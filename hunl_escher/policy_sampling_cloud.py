"""Separate cloud jobs for local sampler testing and gated Slumbot follow-up."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import threading

from .checkpointing import sha256_file
from .policy_replay_artifacts import prepare, safe_path
from .policy_sampling_experiment import run_fit
from .policy_sampling_evaluate import run_evaluation, verified_refits
from .policy_sampling_slumbot import run_benchmark, verified_gate
from .storage_transfer import run_transfer
from .train import write_json


def storage(*args):
    run_transfer(["gcloud", "storage", *map(str, args)], timeout=1800)


@contextmanager
def publication(workspace, output_uri):
    root = Path(workspace)
    root.mkdir(parents=True, exist_ok=False)
    output = root / "results"
    output.mkdir()
    status = dict(status="STARTING", output_uri=output_uri)
    stop = threading.Event()

    def stage(name):
        status["status"] = name
        write_json(output / "cloud_status.json", status)

    def sync(final=False):
        exclude = r"^cloud_status\.json$|^slumbot_ready\.json$" + (r"|\.jsonl$" if not final else "")
        storage("rsync", "--recursive", "--exclude", exclude, output, output_uri)
        if final and (output / "slumbot_ready.json").exists():
            storage("cp", output / "slumbot_ready.json", output_uri + "/slumbot_ready.json")
        storage("cp", output / "cloud_status.json", output_uri + "/cloud_status.json")

    def loop():
        while not stop.wait(120):
            try:
                sync()
            except Exception as exc:
                print(f"Periodic upload failed; final upload remains mandatory: {exc}", flush=True)

    stage("PREPARING")
    publisher = threading.Thread(target=loop, daemon=True)
    publisher.start()
    try:
        yield output, status, stage
    except BaseException as exc:
        status["error"] = f"{type(exc).__name__}: {exc}"
        stage("FAILED")
        raise
    finally:
        stop.set()
        publisher.join()
        sync(final=True)


def run_local(workspace, source_uri, output_uri, expected_policy_sha256, *, iteration=1658):
    source_uri, output_uri = source_uri.rstrip("/"), output_uri.rstrip("/")
    if (source_uri == output_uri or output_uri.startswith(source_uri + "/")
            or source_uri.startswith(output_uri + "/")):
        raise ValueError("Use a separate cloud output prefix")
    with publication(workspace, output_uri) as (output, status, stage):
        root = Path(workspace)
        state = root / "policy-replay"
        state.mkdir()
        policy = state / f"policy_{iteration:06d}.pkl"
        status.update(source_uri=source_uri, expected_policy_sha256=expected_policy_sha256, iteration=iteration)
        storage("cp", f"{source_uri}/training/policy_{iteration:06d}.pkl", policy)
        if sha256_file(policy) != expected_policy_sha256:
            raise ValueError("Source policy SHA-256 differs from the requested baseline")
        prepare(f"{source_uri}/training/state_{iteration:06d}", state)
        stage("FITTING")
        run_fit(state, policy, output / "fit", seeds=(1701, 1702), threads=8)
        stage("EVALUATING")
        report = run_evaluation(output / "fit", output / "evaluation", workers=8, max_seconds=7200)
        design, _, hashes = verified_refits(output / "fit")
        write_json(output / "slumbot_ready.json", dict(
            status="READY" if report["slumbot_eligible"] else "NOT_ELIGIBLE",
            experiment_sha256=sha256_file(output / "fit/experiment.json"),
            evaluation_manifest_sha256=sha256_file(output / "evaluation/manifest.json"),
            evaluation_report_sha256=sha256_file(output / "evaluation/report.json"),
            external_fit_seed=design["external_fit_seed"], policies=hashes))
        stage("COMPLETE" if report["status"] == "COMPLETE" else "EVALUATION_" + report["status"])
    return status


def download_followup(source_uri, root, expected_experiment_sha256):
    root = Path(root)
    for name in ("slumbot_ready.json", "fit/experiment.json", "evaluation/manifest.json", "evaluation/report.json"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        storage("cp", source_uri + "/" + name, path)
    ready = json.loads((root / "slumbot_ready.json").read_text())
    if ready["status"] != "READY" or ready["experiment_sha256"] != expected_experiment_sha256:
        raise ValueError("Sampler experiment is not ready for this follow-up")
    for name, key in (("fit/experiment.json", "experiment_sha256"),
                      ("evaluation/manifest.json", "evaluation_manifest_sha256"),
                      ("evaluation/report.json", "evaluation_report_sha256")):
        if sha256_file(root / name) != ready[key]:
            raise ValueError("Follow-up input checksum mismatch")
    design = json.loads((root / "fit/experiment.json").read_text())
    for row in design["results"]:
        path = safe_path(root / "fit", row["checkpoint"])
        path.parent.mkdir(parents=True, exist_ok=True)
        storage("cp", source_uri + "/fit/" + row["checkpoint"], path)
    design, _, hashes = verified_refits(root / "fit")
    if hashes != ready["policies"] or design["external_fit_seed"] != ready["external_fit_seed"]:
        raise ValueError("Follow-up policy identities differ")
    verified_gate(root / "fit", root / "evaluation")
    return root


def run_slumbot(workspace, source_uri, output_uri, expected_experiment_sha256):
    source_uri, output_uri = source_uri.rstrip("/"), output_uri.rstrip("/")
    if output_uri != source_uri + "/slumbot":
        raise ValueError("Follow-up output must be the separate slumbot subdirectory")
    with publication(workspace, output_uri) as (output, status, stage):
        root = download_followup(source_uri, Path(workspace) / "local-experiment", expected_experiment_sha256)
        stage("SLUMBOT")
        report = run_benchmark(root / "fit", root / "evaluation", output / "benchmark", max_seconds=86400)
        stage("COMPLETE" if report["status"] == "COMPLETE" else "SLUMBOT_" + report["status"])
    return status


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--role", choices=("local", "slumbot"), default="local")
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--source-uri", required=True)
    p.add_argument("--output-uri", required=True)
    p.add_argument("--expected-policy-sha256")
    p.add_argument("--expected-experiment-sha256")
    p.add_argument("--iteration", type=int, default=1658)
    args = vars(p.parse_args())
    role = args.pop("role")
    if role == "local":
        args.pop("expected_experiment_sha256")
        if not args["expected_policy_sha256"]:
            p.error("Local role requires --expected-policy-sha256")
        result = run_local(**args)
    else:
        args.pop("expected_policy_sha256")
        args.pop("iteration")
        if not args["expected_experiment_sha256"]:
            p.error("Slumbot role requires --expected-experiment-sha256")
        result = run_slumbot(**args)
    if result["status"] != "COMPLETE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
