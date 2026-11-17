"""Frozen-replay A/B experiment: uniform groups versus objective-mass sampling."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import random
import time
from types import SimpleNamespace

import numpy as np
import torch

from unbiased_escher.holdem_fitting import fit_validated_policy, policy_validation_mask
from unbiased_escher.policy_sampling import PolicyGroupSampler
from unbiased_escher.replay_grouping import grouped_policy_data, CHUNK_ROWS
from .checkpointing import LoadedHUNLPolicy, sha256_file
from .game import load_hunl_game, serialisable_game_definition
from .policy_replay_artifacts import load_policy_replay, prepare
from .train import write_json

ARMS = ("uniform", "objective_mass")


def model_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def cells_for_groups(features, encoder):
    player, street = encoder.replay_context_columns()
    cells = np.empty(len(features), dtype=np.int8)
    for start in range(0, len(features), CHUNK_ROWS):
        end = min(start + CHUNK_ROWS, len(features))
        rows = features.numpy_batch(start, end)
        cells[start:end] = 4 * rows[:, player].argmax(1) + rows[:, street].argmax(1)
    return cells


def fit_metrics(model, data, ids, cells, batch_size):
    """Per-street weighted KL separates fitting error from target entropy."""
    features, targets, masks, weights = data
    sums = np.zeros((8, 4), dtype=np.float64)  # mass, CE, entropy, L1
    model.eval()
    with torch.no_grad():
        for start in range(0, len(ids), batch_size):
            selected = ids[start:start + batch_size]
            idx = torch.as_tensor(selected)
            y = targets[idx].double()
            logp = torch.log_softmax(model(features[idx]).masked_fill(masks[idx] != 1, -1e20), -1).double()
            ce = -(y * logp).sum(-1)
            entropy = -(y * y.clamp_min(1e-30).log()).sum(-1)
            l1 = (logp.exp() - y).abs().sum(-1)
            rows = torch.stack((torch.ones_like(ce), ce, entropy, l1), -1) * weights[idx, None].double()
            np.add.at(sums, cells[selected], rows.numpy())

    def report(row):
        if row[0] == 0:
            return None
        _, ce, entropy, l1 = row / row[0]
        return dict(weight_mass=float(row[0]), weighted_ce=float(ce), target_entropy=float(entropy),
                    weighted_kl=float(ce - entropy), weighted_action_l1=float(l1))
    return dict(overall=report(sums.sum(0)),
                by_street={name: report(sums.reshape(2, 4, 4)[:, i].sum(0))
                           for i, name in enumerate(("preflop", "flop", "turn", "river"))})


def sampler_diagnostics(data, train_ids, cells, batch_size, seed, batches=256):
    weights = data[3]
    result = {}
    saved_rng = random.getstate()
    try:
        for arm in ARMS:
            random.seed(seed)
            sampler = PolicyGroupSampler(train_ids, weights, batch_size, mode=arm, seed=seed)
            counts, masses = [], []
            for _ in range(batches):
                idx, factors = sampler.sample()
                pre = torch.as_tensor(cells[idx.numpy()] % 4 == 0)
                counts.append(int(pre.sum()))
                masses.append(float((torch.ones(len(idx)) * factors)[pre].sum() / len(idx)))
            result[arm] = dict(batches=batches, preflop_draws_mean=float(np.mean(counts)),
                preflop_draws_sd=float(np.std(counts)), batches_without_preflop=int(np.count_nonzero(np.asarray(counts) == 0)),
                preflop_objective_mass_per_batch_mean=float(np.mean(masses)),
                preflop_objective_mass_per_batch_sd=float(np.std(masses)))
    finally:
        random.setstate(saved_rng)
    return result


def run_fit(state, policy, output, *, seeds=(1701, 1702), threads=8,
            smoke=False, smoke_updates=8):
    if threads < 1 or not seeds or len(set(seeds)) != len(seeds) or any(s < 0 for s in seeds):
        raise ValueError("Invalid thread count or distinct fit seeds")
    if smoke_updates < 1:
        raise ValueError("Invalid smoke update count")
    root = Path(output)
    if root.exists():
        raise FileExistsError("Use a new output directory; existing experiments are immutable")
    torch.set_num_threads(threads)
    buffer, metadata, manifest = load_policy_replay(state)
    loaded = LoadedHUNLPolicy(load_hunl_game(metadata["config"]["action_menu"]), policy)
    payload, config = loaded.checkpoint, metadata["config"]
    iteration = metadata["solver"]["num_iteration"]
    if (metadata["game"] != serialisable_game_definition(config["action_menu"]) or metadata["encoder"] != payload["feature_encoder"]
            or iteration != payload["outer_iteration"] or config != payload["training_config"]):
        raise ValueError("Policy and replay checkpoint identities differ")
    saved_weights = metadata["average_policy_trainer"]["model"]
    if (saved_weights.keys() != loaded.model.state_dict().keys() or
            any(not torch.equal(saved_weights[k], v) for k, v in loaded.model.state_dict().items())):
        raise ValueError("Source policy does not match the full checkpoint's policy weights")
    if buffer.infostate_buf.shape[1] != payload["input_size"] or buffer.action_size != payload["num_actions"]:
        raise ValueError("Replay input width differs from policy")
    if not config["average_policy_reset_each_fit"]:
        raise ValueError("This experiment requires the source's reset-each-fit policy contract")
    if not hasattr(loaded.model, "reset_parameters"):
        raise ValueError("Source policy model cannot be reinitialised")
    root.mkdir(parents=True)
    updates = smoke_updates if smoke else int(config["average_policy_train_steps"])
    batch_size = int(config["ave_policy_batch_size"])
    fraction = float(config["policy_validation_fraction"])
    if not 0 < fraction < 1 or updates < 1:
        raise ValueError("Invalid source policy fitting configuration")
    design = dict(status="FITTING", source_state=str(Path(state).resolve()),
        source_manifest_sha256=sha256_file(Path(state) / "manifest.json"),
        source_policy=str(Path(policy).resolve()), source_policy_sha256=sha256_file(policy),
        source_iteration=iteration, source_training_commit=metadata["repository_commit"],
        source_config=config, network=payload["policy_model"],
        fit_seeds=list(seeds), arms=list(ARMS), smoke=smoke, updates=updates,
        learning_rate=config["average_policy_learning_rate"], batch_size=batch_size,
        threads=threads, validation_fraction=fraction, results=[],
        only_intervention="group sampling and mathematically required loss compensation",
        inference_scope="Frozen P32-style replay at one training seed; not a new self-play run",
        validation_note="Used for update selection; not an independent confirmatory test",
        external_fit_seed=int(seeds[0]),
        evaluation_design="All fit seeds retained for primary mean duplicate-deal head-to-head, alpha=.05. Fresh pilot excluded. Only after primary improvement, separately run Slumbot for the first predeclared fit seed, both arms with fresh independent hands. No seed selection by performance.")
    from unbiased_escher import holdem_fitting, policy_sampling, replay_grouping
    design["implementation_sha256"] = {Path(path).name: sha256_file(path) for path in
        (__file__, holdem_fitting.__file__, policy_sampling.__file__, replay_grouping.__file__)}
    design["versions"] = dict(numpy=np.__version__, torch=torch.__version__)
    write_json(root / "experiment.json", design)
    started = time.monotonic()
    try:
        data, coverage = grouped_policy_data(buffer, iteration, config["gamma"], "cpu",
            context_columns=loaded.feature_encoder.replay_context_columns())
        heldout = policy_validation_mask(data[0], fraction)
        train_ids, valid_ids = np.flatnonzero(~heldout), np.flatnonzero(heldout)
        if not len(train_ids) or not len(valid_ids):
            raise ValueError("Replay must contain both training and validation information sets")
        cells = cells_for_groups(data[0], loaded.feature_encoder)
        design["coverage"] = coverage
        design["sampler_diagnostics"] = sampler_diagnostics(data, train_ids, cells, batch_size, seeds[0])
        write_json(root / "experiment.json", design)
        for seed in seeds:
            torch.manual_seed(seed)
            initial = deepcopy(loaded.model)
            initial.reset_parameters()
            initial_hash = model_hash(initial)
            # Alternate order across repetitions to reduce systematic thermal/order bias.
            arms = ARMS if seeds.index(seed) % 2 == 0 else ARMS[::-1]
            for arm in arms:
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                model = deepcopy(initial)
                trainer = SimpleNamespace(model=model,
                    optimizer=torch.optim.Adam(model.parameters(), lr=config["average_policy_learning_rate"]),
                    _grouped_training_data=lambda _: data, validation_fraction=fraction,
                    validation_steps=config["policy_validation_steps"], train_steps=updates,
                    batch_size=batch_size, policy_group_sampling=arm, policy_sampling_seed=seed)
                tick = time.monotonic()
                fit_validated_policy(trainer, iteration)
                fit_seconds = time.monotonic() - tick
                result = dict(arm=arm, fit_seed=seed, initial_model_sha256=initial_hash,
                    fit_seconds=fit_seconds, fit=trainer.last_fit_diagnostics,
                    training=fit_metrics(model, data, train_ids, cells, batch_size),
                    validation=fit_metrics(model, data, valid_ids, cells, batch_size))
                checkpoint = deepcopy(payload)
                checkpoint["policy_state_dict"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                checkpoint["experiment_id"] = "frozen_policy_sampler_ablation"
                checkpoint["experiment_name"] = "Frozen replay policy sampler A/B"
                checkpoint["created_utc"] = datetime.now(timezone.utc).isoformat()
                checkpoint["checkpoint_kind"] = "frozen_replay_policy_refit"
                checkpoint["policy_refit"] = dict(source_sha256=design["source_policy_sha256"],
                    arm=arm, fit_seed=seed, initial_model_sha256=initial_hash,
                    fit_seconds=fit_seconds, smoke=smoke, fit=trainer.last_fit_diagnostics)
                path = root / f"{arm}_seed{seed}.pkl"
                with path.open("xb") as handle:
                    pickle.dump(checkpoint, handle, protocol=pickle.HIGHEST_PROTOCOL)
                result.update(checkpoint=path.name, sha256=sha256_file(path))
                design["results"].append(result)
                write_json(root / "experiment.json", design)
                print(json.dumps({k: result[k] for k in ("arm", "fit_seed", "fit_seconds", "validation")}), flush=True)
        design["status"] = "COMPLETE"
    except BaseException as exc:
        design.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        design["elapsed_seconds"] = time.monotonic() - started
        write_json(root / "experiment.json", design)
    return design


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="Download only policy replay from a trusted GCS checkpoint")
    prep.add_argument("--source-state", required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--metadata-only", action="store_true")
    fit = sub.add_parser("fit")
    fit.add_argument("--state", type=Path, required=True)
    fit.add_argument("--policy", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--seeds", type=int, nargs="+", default=[1701, 1702])
    fit.add_argument("--threads", type=int, default=8)
    fit.add_argument("--smoke", action="store_true")
    fit.add_argument("--smoke-updates", type=int, default=8)
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    if command == "prepare":
        args["source"] = args.pop("source_state")
        prepare(**args)
    else:
        run_fit(**args)


if __name__ == "__main__":
    main()
