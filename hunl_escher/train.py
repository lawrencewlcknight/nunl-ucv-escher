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

from unbiased_escher.replay_grouping import reservoir_counts, learner_replay_allocation
from importlib.metadata import version

from .config import baseline_config, smoke_config, make_solver
from .game import serialisable_game_definition
from .checkpointing import save_policy_checkpoint
from .checkpoint_schedule import CheckpointSchedule
from .recovery_retention import prune_recovery_checkpoints
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
            "policy_replay_rows": min(solver.ave_policy_trainer.buffer.cur_id,
                                      solver.ave_policy_trainer.buffer.buffer_size),
            "policy_replay": reservoir_counts(solver.ave_policy_trainer.buffer),
            "allocated_learner_replay_bytes": learner_replay_allocation(solver),
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
    parser.add_argument("--additional-training-seconds", type=float,
                        help="Stop and export after this many MORE active seconds, at an iteration boundary")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--policy-every", type=int, default=None,
                        help="Iterations between policy extractions (smoke: 1; otherwise: disabled); also fit at final endpoint")
    parser.add_argument("--checkpoint-every", type=int, default=0,
                        help="Iterations between recovery saves, without policy fitting (0: disabled)")
    parser.add_argument("--recovery-seconds", "--checkpoint-seconds", dest="recovery_seconds",
                        type=float, default=21600,
                        help="Wall seconds between recovery checks (default: 21600); skip if saved within the last hour (0: disabled)")
    parser.add_argument("--policy-seconds", type=float, default=21600,
                        help="Active learner seconds between policy fits, excluding policy fitting and persistence")
    args = parser.parse_args(argv)
    if min(args.iterations, args.workers, args.threads) < 1:
        parser.error("iterations, workers and threads must be positive")
    if args.config and args.smoke:
        parser.error("Choose either a custom config or smoke")
    if args.additional_training_seconds is not None and (
            not math.isfinite(args.additional_training_seconds) or args.additional_training_seconds <= 0):
        parser.error("additional-training-seconds must be finite and positive")
    policy_every = args.policy_every if args.policy_every is not None else (1 if args.smoke else 0)
    if policy_every < 0 or args.checkpoint_every < 0:
        parser.error("policy-every and checkpoint-every must be nonnegative")
    if not math.isfinite(args.recovery_seconds) or args.recovery_seconds < 0:
        parser.error("recovery-seconds must be finite and nonnegative")
    if not math.isfinite(args.policy_seconds) or args.policy_seconds <= 0:
        parser.error("policy-seconds must be finite and positive")
    config = (json.loads(args.config.read_text()) if args.config else
              smoke_config() if args.smoke else baseline_config())
    contract = {"config": config, "seed": args.seed, "workers": args.workers,
                "threads": args.threads, "policy_every": policy_every,
                "checkpoint_every": args.checkpoint_every,
                "recovery_seconds": args.recovery_seconds, "policy_seconds": args.policy_seconds,
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
        schedule = CheckpointSchedule()
        if args.resume:
            pointer = json.loads(latest.read_text())
            payload = read_training_state(root / pointer["state"])
            restore_training_state(solver, payload, **kwargs)
            schedule = CheckpointSchedule(**pointer.get("checkpoint_schedule", payload["checkpoint_schedule"]))
            del payload  # Do not retain restored replay mappings throughout training.
        stage_started_active = solver._training_elapsed_seconds()
        while solver.num_iteration < args.iterations:
            started = time.perf_counter()
            solver.iteration()
            learner_seconds = time.perf_counter() - started
            solver._resume_training_elapsed_seconds += learner_seconds
            iteration = solver.num_iteration
            active_seconds = solver._training_elapsed_seconds()
            budget_reached = (args.additional_training_seconds is not None and
                              active_seconds - stage_started_active >= args.additional_training_seconds)
            previous_recovery_check = schedule.last_recovery_check_wall_seconds
            recovery, policy_fit = schedule.due(
                iteration=iteration, final=iteration == args.iterations or budget_reached,
                active_seconds=active_seconds, wall_seconds=time.time(),
                policy_every=policy_every, checkpoint_every=args.checkpoint_every,
                recovery_seconds=args.recovery_seconds, policy_seconds=args.policy_seconds)
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
                   "policy_validation": getattr(solver.ave_policy_trainer, "last_fit_diagnostics", {}) if policy_fit else None,
                   "policy_replay_coverage": getattr(solver.ave_policy_trainer, "last_replay_diagnostics", {}) if policy_fit else None,
                   "regret_cache": [getattr(t, "last_target_cache_stats", {}) for t in solver.regret_trainers],
                   "critic_target_versions": [m.target_version for m in solver.q_value_trainer.members],
                   "critic_updates_per_member": [m.last_fit_stats["updates"] for m in solver.q_value_trainer.members],
                   "critic_max_updates_per_member": config["baseline_network_train_steps"],
                   **learner_metrics(solver, learner_seconds),
                   "critic_cache": [getattr(m, "last_target_cache_stats", {}) for m in solver.q_value_trainer.members]}
            persistence_started = time.perf_counter()
            if recovery:
                state_name = f"state_{iteration:06d}"
                schedule.completed(active_seconds, wall_seconds=time.time(),
                                   recovery=recovery, policy=policy_fit)
                payload = build_training_state(
                    solver, **kwargs, checkpoint_id=str(iteration), captured_checkpoints=[])
                payload["checkpoint_schedule"] = schedule.state()
                save_training_state(root / state_name, payload)
                # Use successful save completion for the one-hour suppression
                # window. The small latest pointer carries this precise time.
                schedule.completed(active_seconds, wall_seconds=time.time(),
                                   recovery=True, policy=policy_fit)
                if policy_fit:
                    save_policy_checkpoint(solver, root / f"policy_{iteration:06d}.pkl",
                                           seed=args.seed, config=config, checkpoint_row=row)
                write_json(latest, {"iteration": iteration, "state": state_name,
                                    "checkpoint_schedule": schedule.state()})
                row["recovery_checkpoints_deleted"] = prune_recovery_checkpoints(root, state_name)
            elif schedule.last_recovery_check_wall_seconds != previous_recovery_check and latest.exists():
                # Persist a skipped check without rewriting replay or changing
                # which training state is resumable. This survives a restart
                # after the recent-save window has expired.
                pointer = json.loads(latest.read_text())
                pointer["checkpoint_schedule"] = schedule.state()
                write_json(latest, pointer)
            row["persistence_seconds"] = time.perf_counter() - persistence_started
            row["iteration_wall_seconds"] = time.perf_counter() - started
            write_json(root / f"metrics_{iteration:06d}.json", row)
            print(json.dumps(row), flush=True)
            if budget_reached:
                break
    finally:
        if hasattr(solver, "close"):
            solver.close()


if __name__ == "__main__":
    main()
