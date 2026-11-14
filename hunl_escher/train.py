"""Bounded local training and resumable smoke runs; never submits cloud jobs."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import math
import time

import numpy as np
import torch
from importlib.metadata import version

from .config import baseline_config, smoke_config, make_solver
from .game import serialisable_game_definition
from .checkpointing import save_policy_checkpoint
from .training_state import build_training_state, restore_training_state, save_training_state, read_training_state


def source_fingerprint():
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for package in ("hunl_escher", "unbiased_escher", "adaptive_escher", "vr_deep_cfr"):
        for path in sorted((root / package).rglob("*")):
            if path.suffix in (".py", ".json"):
                digest.update(str(path.relative_to(root)).encode())
                digest.update(path.read_bytes())
    return digest.hexdigest()


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)) + "\n")
    temp.replace(path)


def checkpoint_due(iteration, final_iteration, *, policy_every, checkpoint_every,
                   checkpoint_seconds, seconds_since_save):
    """Recovery never requires policy extraction; final output includes both."""
    policy = iteration % policy_every == 0 or iteration == final_iteration
    recovery = (policy
                or (checkpoint_every > 0 and iteration % checkpoint_every == 0)
                or (checkpoint_seconds > 0 and seconds_since_save >= checkpoint_seconds))
    return policy, bool(recovery)


def learner_metrics(solver, iteration_seconds):
    regret = [dict(t.last_fit_stats) for t in solver.regret_trainers]
    critics = [dict(m.last_fit_stats) for m in solver.q_value_trainer.members]
    calibration = (dict(solver.calibration_trainer.last_fit_stats)
                   if solver.calibration_trainer is not None else None)
    fits = regret + critics + ([calibration] if calibration else [])
    # Fits are sequential in this production configuration. Cache construction
    # is already INCLUDED in critic seconds; never add it again.
    fit_seconds = sum(fit["seconds"] for fit in fits)
    return {"regret_fits": regret, "critic_fits": critics, "calibration_fit": calibration,
            "collection_and_overhead_seconds": max(0.0, iteration_seconds - fit_seconds),
            "regret_rows_by_player_street": [
                [int(solver._architecture_stats[f"regret_rows_player_{p}_street_{s}"])
                 for s in range(4)] for p in range(2)],
            "policy_replay_rows": len(solver.ave_policy_trainer.buffer),
            "minimum_sample_probability": solver._minimum_sample_probability}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--iterations", type=int, default=2, help="Total completed iterations, including restored ones")
    parser.add_argument("--workers", type=int, default=1, help="Use 8 for the inherited production collection layout")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--policy-every", type=int, default=None,
                        help="Iterations between policy extractions (smoke: 1; otherwise: 50); also fit at final endpoint")
    parser.add_argument("--checkpoint-every", type=int, default=0,
                        help="Iterations between recovery saves, without policy fitting (0: disabled)")
    parser.add_argument("--checkpoint-seconds", type=float, default=1800,
                        help="Recovery interval in wall seconds, checked between iterations (0: disabled)")
    args = parser.parse_args(argv)
    if min(args.iterations, args.workers, args.threads) < 1:
        parser.error("iterations, workers and threads must be positive")
    if args.config and args.smoke:
        parser.error("Choose either a custom config or smoke")
    policy_every = args.policy_every if args.policy_every is not None else (1 if args.smoke else 50)
    if policy_every < 1 or args.checkpoint_every < 0:
        parser.error("policy-every must be positive and checkpoint-every nonnegative")
    if not math.isfinite(args.checkpoint_seconds) or args.checkpoint_seconds < 0:
        parser.error("checkpoint-seconds must be finite and nonnegative")
    config = (json.loads(args.config.read_text()) if args.config else
              smoke_config() if args.smoke else baseline_config())
    contract = {"config": config, "seed": args.seed, "workers": args.workers,
                "threads": args.threads, "policy_every": policy_every,
                "checkpoint_every": args.checkpoint_every,
                "checkpoint_seconds": args.checkpoint_seconds,
                "source_fingerprint": source_fingerprint(),
                "game": serialisable_game_definition(config["action_menu"]), "smoke": args.smoke,
                "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                            "numpy": np.__version__, "open_spiel": version("open_spiel"),
                            "ray": version("ray") if args.workers > 1 else None}}
    root = args.output
    manifest = root / "manifest.json"
    if args.resume:
        if not manifest.exists() or json.loads(manifest.read_text()) != contract:
            raise ValueError("Resume requires the same source, configuration and runtime")
    else:
        if root.exists() and any(root.iterdir()):
            raise ValueError("Use an empty output directory or --resume")
        root.mkdir(parents=True, exist_ok=True)
        write_json(manifest, contract)
    torch.set_num_threads(args.threads)
    solver = make_solver(config, seed=args.seed, workers=args.workers)
    try:
        kwargs = dict(seed=args.seed, repository_commit=contract["source_fingerprint"], config=config)
        latest = root / "latest.json"
        if args.resume:
            pointer = json.loads(latest.read_text())
            restore_training_state(solver, read_training_state(root / pointer["state"]), **kwargs)
        last_save_time = time.perf_counter()
        while solver.num_iteration < args.iterations:
            started = time.perf_counter()
            solver.iteration()
            learner_seconds = time.perf_counter() - started
            solver._resume_training_elapsed_seconds += learner_seconds
            iteration = solver.num_iteration
            policy_fit, recovery = checkpoint_due(
                iteration, args.iterations, policy_every=policy_every,
                checkpoint_every=args.checkpoint_every,
                checkpoint_seconds=args.checkpoint_seconds,
                seconds_since_save=time.perf_counter() - last_save_time)
            policy_seconds = 0.0
            if policy_fit:
                rng = solver._capture_rng_state()
                started_policy = time.perf_counter()
                try:
                    solver.train_average_policy()
                finally:
                    solver._restore_rng_state(rng)
                policy_seconds = time.perf_counter() - started_policy
            row = {"iteration": iteration, "nodes": solver.nodes_touched, "trajectories": solver.episode,
                   "learner_seconds": learner_seconds, "policy_fit_seconds": policy_seconds,
                   "learning_seconds": learner_seconds + policy_seconds,
                   "training_elapsed_seconds": solver._training_elapsed_seconds(),
                   "policy_fitted": policy_fit, "recovery_saved": recovery,
                   "policy_loss": solver._last_average_policy_loss if policy_fit else None,
                   "critic_target_versions": [m.target_version for m in solver.q_value_trainer.members],
                   "critic_updates_per_member": [m.last_fit_stats["updates"] for m in solver.q_value_trainer.members],
                   "critic_max_updates_per_member": config["baseline_network_train_steps"],
                   **learner_metrics(solver, learner_seconds),
                   "critic_cache": [getattr(m, "last_target_cache_stats", {}) for m in solver.q_value_trainer.members]}
            persistence_started = time.perf_counter()
            if recovery:
                state_name = f"state_{iteration:06d}"
                save_training_state(root / state_name, build_training_state(
                    solver, **kwargs, checkpoint_id=str(iteration), captured_checkpoints=[]))
                if policy_fit:
                    save_policy_checkpoint(solver, root / f"policy_{iteration:06d}.pkl",
                                           seed=args.seed, config=config, checkpoint_row=row)
                write_json(latest, {"iteration": iteration, "state": state_name})
                last_save_time = time.perf_counter()
            row["persistence_seconds"] = time.perf_counter() - persistence_started
            row["iteration_wall_seconds"] = time.perf_counter() - started
            write_json(root / f"metrics_{iteration:06d}.json", row)
            print(json.dumps(row), flush=True)
    finally:
        if hasattr(solver, "close"):
            solver.close()


if __name__ == "__main__":
    main()
