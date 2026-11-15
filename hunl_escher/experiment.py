"""Four hours + restart + four hours, then fixed-policy evaluations. No cloud submission."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import re

from .checkpointing import load_checkpoint_payload, sha256_file
from .experiment_runtime import run_child
from .preflight import preflight
from .train import source_fingerprint, write_json


def publish_policy(training, output, label):
    pointer = json.loads((training / "latest.json").read_text())
    source = training / f"policy_{pointer['iteration']:06d}.pkl"
    payload = load_checkpoint_payload(source)
    temporary = output / f"{label}.pkl.tmp"
    shutil.copyfile(source, temporary)
    temporary.replace(output / f"{label}.pkl")
    row = json.loads((training / f"metrics_{pointer['iteration']:06d}.json").read_text())
    record = dict(label=label, sha256=sha256_file(source), outer_iteration=payload["outer_iteration"],
                  active_seconds=payload["training_elapsed_seconds"], metrics=row,
                  recovery=pointer["state"])
    write_json(output / f"{label}.json", record)  # Marker published after complete policy.
    return record


def upload_recovery(root, uri):
    """Publish a verified local recovery, then prune the previous cloud recovery."""
    training = root / "training"
    pointer = json.loads((training / "latest.json").read_text())
    remote = uri.rstrip("/")
    subprocess.run(["gcloud", "storage", "rsync", "--recursive", str(training / pointer["state"]),
                    remote+"/training/"+pointer["state"]], check=True, timeout=1200)
    for name in ("manifest.json", "latest.json"):
        subprocess.run(["gcloud", "storage", "cp", str(training / name), remote+"/training/"+name], check=True, timeout=120)
    publication = root / "cloud_checkpoint.json"
    previous = json.loads(publication.read_text())["state"] if publication.exists() else None
    if previous and previous != pointer["state"]:
        if not re.fullmatch(r"state_[0-9]+", previous):
            raise ValueError("Invalid previous cloud checkpoint directory")
        subprocess.run(["gcloud", "storage", "rm", "--recursive", remote+"/training/"+previous],
                       check=True, timeout=300)
    write_json(publication, {"state": pointer["state"]})


def upload_checkpoint(root, uri, label):
    """Recovery upload completes before publishing a cloud policy marker."""
    upload_recovery(root, uri)
    for suffix in ("pkl", "json"):
        subprocess.run(["gcloud", "storage", "cp", str(root / f"{label}.{suffix}"),
                        uri.rstrip("/")+f"/{label}.{suffix}"], check=True, timeout=120)


def training_diagnostics(root):
    """Engineering evidence, explicitly separate from playing-strength evidence."""
    import numpy as np
    import torch
    rows = [json.loads(path.read_text()) for path in sorted((root / "training").glob("metrics_*.json"))]
    def finite_numbers(value):
        if isinstance(value, float):
            return math.isfinite(value)
        if isinstance(value, dict):
            return all(finite_numbers(v) for v in value.values())
        if isinstance(value, list):
            return all(finite_numbers(v) for v in value)
        return True
    early, late = [load_checkpoint_payload(root / f"{label}.pkl") for label in ("P4", "P8")]
    changed = any(not torch.equal(value, late["policy_state_dict"][key])
                  for key, value in early["policy_state_dict"].items())
    policy_finite = all(torch.isfinite(value).all().item() for payload in (early, late)
                        for value in payload["policy_state_dict"].values())
    coverage = np.asarray([row["regret_rows_by_player_street"] for row in rows]).sum(axis=0).tolist()
    stages = {}
    for label in ("P4", "P8"):
        path = root / f"train-{label}.resources.json"
        stages[label] = json.loads(path.read_text())
    elapsed = sum(row["learner_seconds"] for row in rows)
    return dict(finite_recorded_metrics=all(finite_numbers(row) for row in rows),
                finite_exported_policy_weights=policy_finite, exported_policy_changed=changed,
                completed_iterations=rows[-1]["iteration"], trajectories=rows[-1]["trajectories"],
                trajectories_per_active_second=rows[-1]["trajectories"]/elapsed,
                collection_and_overhead_seconds=sum(row["collection_and_overhead_seconds"] for row in rows),
                network_fit_seconds=sum(sum(f["seconds"] for f in row["regret_fits"]+row["critic_fits"]+
                    ([row["calibration_fit"]] if row["calibration_fit"] else [])) for row in rows),
                regret_rows_by_player_street=coverage,
                critic_target_versions=rows[-1]["critic_target_versions"],
                policy_fit_seconds=sum(row["policy_fit_seconds"] for row in rows),
                persistence_seconds=sum(row["persistence_seconds"] for row in rows),
                policy_replay=rows[-1]["policy_replay"], resource_measurements=stages,
                interpretation="Engineering diagnostics do not establish strategic improvement")


def run_experiment(output, *, stage_seconds=14400, workers=8, threads=4, seed=20261115,
                   smoke=False, skip_preflight=False, local_workers=8,
                   training_wall_seconds=43200, evaluation_seconds=7200, upload_uri=None):
    if not all(math.isfinite(x) and x > 0 for x in (stage_seconds, training_wall_seconds, evaluation_seconds)):
        raise ValueError("Time budgets must be finite and positive")
    if min(workers, threads, local_workers) < 1:
        raise ValueError("Worker/thread counts must be positive")
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = dict(status="STARTING", stage_seconds=stage_seconds, seed=seed, smoke=smoke,
                  source_fingerprint=source_fingerprint(), checkpoints={}, evaluations={})
    write_json(root / "experiment.json", report)
    try:
        if not smoke and not skip_preflight:
            selected = preflight(root / "preflight")
            workers, threads = selected["workers"], selected["threads"]
        if smoke:
            workers = threads = local_workers = 1
        report.update(workers=workers, threads=threads, status="TRAINING")
        write_json(root / "experiment.json", report)
        training = root / "training"
        common = [sys.executable, "-m", "hunl_escher.train", "--output", str(training),
                  "--iterations", "1000000", "--workers", str(workers), "--threads", str(threads),
                  "--seed", str(seed), "--policy-every", "0", "--policy-seconds", "1000000000",
                  "--additional-training-seconds", str(stage_seconds)]
        if smoke:
            common.append("--smoke")
        for label in ("P4", "P8"):
            remaining = training_wall_seconds-(time.monotonic()-started)
            if remaining <= 0:
                raise TimeoutError("Training wall budget exhausted")
            result = run_child([*common, *(["--resume"] if label == "P8" else [])],
                               root / f"train-{label}.log", timeout=remaining)
            if result["returncode"] != 0:
                raise RuntimeError(f"Training stage {label} failed or exceeded wall limit; see its log")
            checkpoint = publish_policy(training, root, label)
            previous = report["checkpoints"].get("P4", {}).get("active_seconds", 0) if label == "P8" else 0
            if checkpoint["active_seconds"]-previous < stage_seconds:
                raise RuntimeError("Iteration ceiling reached before the requested active training budget")
            report["checkpoints"][label] = checkpoint
            write_json(root / "experiment.json", report)
            if upload_uri:
                upload_checkpoint(root, upload_uri, label)
        report["training_diagnostics"] = training_diagnostics(root)
        # One primary comparison. Reference matches are descriptive diagnostics.
        primary_limits = ["--pilot-pairs", "4", "--minimum", "8", "--maximum", "8"] if smoke else []
        matches = [("primary", "P8", str(root / "P4.pkl"), primary_limits)]
        for label in ("P4", "P8"):
            for opponent in ("random", "check_call"):
                n = "8" if smoke else "10000"
                matches.append((f"{label}-{opponent}", label, opponent,
                                ["--pilot-pairs", "4" if smoke else "1000", "--minimum", n, "--maximum", n]))
        evaluation_deadline = time.monotonic()+evaluation_seconds
        for index, (name, label, opponent, limits) in enumerate(matches):
            remaining = evaluation_deadline-time.monotonic()
            if remaining <= 1:
                report["evaluations"][name] = {"status": "SKIPPED_TIME_LIMIT"}
                continue
            # Reserve a fifth of local wall budget for the four reference matches.
            budget = remaining*.8 if name == "primary" else remaining/(len(matches)-index)
            directory = root / name
            outcome = run_child([sys.executable, "-m", "hunl_escher.evaluate",
                      "--a", str(root / f"{label}.pkl"), "--b", opponent,
                      "--output", str(directory), "--workers", str(local_workers),
                      "--seed", str(8301 if name == "primary" else 8401),
                      "--max-seconds", str(budget), *limits],
                      root / f"eval-{name}.log", timeout=budget+60)
            path = directory / "report.json"
            report["evaluations"][name] = (json.loads(path.read_text()) if path.exists() else
                dict(status="INCOMPLETE_TIME_LIMIT" if outcome["timed_out"] else "EVALUATION_ERROR",
                     returncode=outcome["returncode"]))
            if name != "primary":
                report["evaluations"][name]["interpretation"] = "descriptive reference; not an additional primary test"
            write_json(root / "experiment.json", report)
        report["status"] = "TRAINING_COMPLETE"
    except Exception as exc:
        report.update(status="FAILED", error_type=type(exc).__name__)
        write_json(root / "training_failed.json", {"error_type": type(exc).__name__})
        raise
    finally:
        report["wall_seconds"] = time.monotonic()-started
        write_json(root / "experiment.json", report)
    from .experiment_report import render
    render(root)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--stage-seconds", type=float, default=14400)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--local-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=20261115)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--skip-preflight", action="store_true")
    p.add_argument("--training-wall-seconds", type=float, default=43200)
    p.add_argument("--evaluation-seconds", type=float, default=7200)
    p.add_argument("--upload-uri", help="Optional gs:// run prefix; publishes checkpoint artifacts")
    args = p.parse_args(argv)
    print(json.dumps(run_experiment(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
