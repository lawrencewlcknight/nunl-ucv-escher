"""Inference-only GCP worker for the corrected P32 GTO Wizard benchmark."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import threading

from .checkpointing import sha256_file
from .gtow_p32 import run_benchmark, verified_candidate
from .policy_replay_artifacts import safe_path
from .storage_transfer import run_transfer
from .train import write_json


def storage(*args):
    # Bounded idempotent storage retries only; never retry live game actions.
    run_transfer(["gcloud", "storage", *map(str, args)], timeout=120)


def snapshot_results(output, snapshot):
    """Copy atomic reports and a fixed, complete-line prefix of each journal."""
    output, snapshot = Path(output), Path(snapshot)
    # Capture status/reports before journals so appended actions cannot leave a
    # newer report paired with an older journal prefix in this snapshot.
    paths = sorted(output.rglob("*.json"), key=lambda p: (p.name != "cloud_status.json", str(p)))
    paths += sorted(output.rglob("*.jsonl"))
    for source in paths:
        target = snapshot / source.relative_to(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as reader, target.open("wb") as writer:
            remaining = os.fstat(reader.fileno()).st_size
            last_newline = 0
            while remaining:
                chunk = reader.read(min(remaining, 1024 * 1024))
                if not chunk:
                    break
                offset = chunk.rfind(b"\n")
                if offset >= 0:
                    last_newline = writer.tell() + offset + 1
                writer.write(chunk)
                remaining -= len(chunk)
            if source.suffix == ".jsonl":
                writer.truncate(last_newline)


@contextmanager
def publication(workspace, output_uri, *, interval=120):
    root = Path(workspace)
    root.mkdir(parents=True, exist_ok=False)
    output = root / "results"
    output.mkdir()
    status = dict(status="STARTING", output_uri=output_uri)
    stop = threading.Event()

    def stage(name):
        status["status"] = name
        write_json(output / "cloud_status.json", status)

    def sync():
        # rsync must never read a file while the benchmark is appending to it.
        with tempfile.TemporaryDirectory(prefix="publication-", dir=root) as directory:
            snapshot = Path(directory)
            snapshot_results(output, snapshot)
            storage("rsync", "--recursive", "--exclude", r"^cloud_status\.json$", snapshot, output_uri)
            storage("cp", snapshot / "cloud_status.json", output_uri + "/cloud_status.json")

    def loop():
        while not stop.wait(interval):
            try:
                sync()
            except Exception as exc:
                print(f"Periodic result upload failed ({type(exc).__name__}); final upload remains mandatory.", flush=True)

    stage("PREPARING")
    publisher = threading.Thread(target=loop, daemon=True)
    publisher.start()
    try:
        yield output, status, stage
    except BaseException as exc:
        status["error_type"] = type(exc).__name__
        stage("INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "FAILED")
        raise
    finally:
        stop.set()
        publisher.join()
        sync()


def download_refit(source_uri, root, expected_experiment_sha256, expected_policy_sha256):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    manifest = root / "experiment.json"
    storage("cp", source_uri.rstrip("/") + "/fit/experiment.json", manifest)
    if sha256_file(manifest) != expected_experiment_sha256:
        raise ValueError("Refit manifest checksum mismatch")
    design = json.loads(manifest.read_text())
    if design.get("status") != "COMPLETE" or design.get("smoke") is not False:
        raise ValueError("Requires a completed production refit")
    # These are four small playable policies, never the 18.7 GB replay.
    for row in design["results"]:
        path = safe_path(root, row["checkpoint"])
        path.parent.mkdir(parents=True, exist_ok=True)
        storage("cp", source_uri.rstrip("/") + "/fit/" + row["checkpoint"], path)
        if sha256_file(path) != row["sha256"]:
            raise ValueError("Refit policy checksum mismatch")
    _, identity = verified_candidate(root)
    if identity["policy_sha256"] != expected_policy_sha256:
        raise ValueError("Corrected P32 candidate checksum mismatch")
    return root


def read_secret(project, secret):
    try:
        result = subprocess.run(["gcloud", "secrets", "versions", "access", "latest",
            "--project", project, "--secret", secret], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        raise RuntimeError("Secret Manager request failed; no API calls made") from None
    key = result.stdout.strip()
    if result.returncode or not key or "\n" in key or "\r" in key:
        # Do not log subprocess output: it may contain the credential.
        raise RuntimeError("Cannot read the GTO API key; check Secret Manager access")
    return key


def run_worker(workspace, source_uri, output_uri, expected_experiment_sha256,
               expected_policy_sha256, project, gto_secret):
    source_uri, output_uri = source_uri.rstrip("/"), output_uri.rstrip("/")
    if (source_uri == output_uri or source_uri.startswith(output_uri + "/")
            or output_uri.startswith(source_uri + "/")):
        raise ValueError("Use a separate benchmark output prefix")
    with publication(workspace, output_uri) as (output, status, stage):
        root = download_refit(source_uri, Path(workspace) / "refit",
                              expected_experiment_sha256, expected_policy_sha256)
        status.update(source_uri=source_uri, policy_sha256=expected_policy_sha256,
                      refit_manifest_sha256=expected_experiment_sha256)
        previous = os.environ.get("GTO_WIZARD_API_KEY")
        try:
            os.environ["GTO_WIZARD_API_KEY"] = read_secret(project, gto_secret)
            # run_benchmark already verifies the policy and authenticated GETs
            # before its first hand; a separate preflight repeats the same work.
            stage("BENCHMARK")
            report = run_benchmark(root, output / "benchmark", max_seconds=86400)
            summary = {key: report[key] for key in ("status", "error_type", "reason",
                "completed_pilot_hands", "completed_confirmation_hands") if key in report}
            status["benchmark"] = summary
            print(json.dumps(dict(phase="benchmark", **summary)), flush=True)
            stage(report["status"])
        finally:
            if previous is None:
                os.environ.pop("GTO_WIZARD_API_KEY", None)
            else:
                os.environ["GTO_WIZARD_API_KEY"] = previous
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--source-uri", required=True)
    parser.add_argument("--output-uri", required=True)
    parser.add_argument("--expected-experiment-sha256", required=True)
    parser.add_argument("--expected-policy-sha256", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--gto-secret", required=True)
    args = vars(parser.parse_args(argv))
    def terminate(signum, frame):
        # Allow the benchmark journal and mandatory final upload to unwind.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise KeyboardInterrupt("Batch worker terminated")
    previous = signal.signal(signal.SIGTERM, terminate)
    try:
        result = run_worker(**args)
    finally:
        signal.signal(signal.SIGTERM, previous)
    if result["status"] != "COMPLETE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
