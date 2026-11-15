"""VM entry point: training publisher or independent external evaluation worker."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .experiment import run_experiment, upload_recovery
from .train import write_json


def sync_output(path, destination, *, exclude=None):
    return subprocess.run(["gcloud", "storage", "rsync", "--recursive",
                          *(["--exclude", exclude] if exclude else []), str(path), destination],
                          timeout=1200, check=False).returncode


def fetch_file(uri, destination):
    temporary = destination.with_suffix(destination.suffix+".download")
    result = subprocess.run(["gcloud", "storage", "cp", uri, str(temporary)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
    if result.returncode == 0:
        temporary.replace(destination)
        return True
    return False


def external_worker(root, uri, *, secret=None, project=None, max_seconds=86400):
    root.mkdir(parents=True, exist_ok=False)
    inputs, outputs = root / "input", root / "benchmarks"
    inputs.mkdir()
    outputs.mkdir()
    if secret:
        # Never print the key, put it in arguments, or write it to disk.
        try:
            key = subprocess.run(["gcloud", "secrets", "versions", "access", "latest",
                                  "--secret", secret, "--project", project],
                                 capture_output=True, text=True, timeout=30)
            if key.returncode == 0 and key.stdout.strip():
                os.environ["GTO_WIZARD_API_KEY"] = key.stdout.strip()
        except (subprocess.TimeoutExpired, OSError):
            pass  # Secret service availability must not prevent Slumbot evaluation.
    jobs, handles = [], []
    deadline = time.monotonic()+max_seconds
    try:
        for provider in ("slumbot", "gtow"):
            handle = (outputs / f"{provider}.log").open("x")
            handles.append(handle)
            jobs.append(subprocess.Popen([sys.executable, "-m", "hunl_escher.benchmark",
                        "--provider", provider, "--experiment", str(inputs),
                        "--output", str(outputs / provider), "--max-seconds", str(max_seconds)],
                        stdout=handle, stderr=subprocess.STDOUT))
        while any(job.poll() is None for job in jobs) and time.monotonic() < deadline:
            for label in ("P4", "P8"):
                for suffix in ("pkl", "json"):
                    path = inputs / f"{label}.{suffix}"
                    if not path.exists():
                        try:
                            fetch_file(uri+f"/{label}.{suffix}", path)
                        except subprocess.TimeoutExpired:
                            pass
            try:
                fetch_file(uri+"/training_failed.json", inputs / "training_failed.json")
                sync_output(outputs, uri+"/benchmarks")
            except subprocess.TimeoutExpired:
                pass
            time.sleep(min(30, max(0, deadline-time.monotonic())))
    finally:
        for job in jobs:
            if job.poll() is None:
                job.terminate()
                try:
                    job.wait(timeout=40)
                except subprocess.TimeoutExpired:
                    job.kill()
                    job.wait()
        for handle in handles:
            handle.close()
        report = {}
        for provider, job in zip(("slumbot", "gtow"), jobs):
            path = outputs / provider / "report.json"
            item = json.loads(path.read_text()) if path.exists() else {"status": "INCOMPLETE_WORKER_EXIT"}
            if job.returncode != 0:
                item["status"] = "INCOMPLETE_WORKER_EXIT"
            report[provider] = {k: v for k, v in item.items() if k not in ("scores", "raw")}
        write_json(outputs / "summary.json", report)
        sync_output(outputs, uri+"/benchmarks")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--role", choices=("train", "external"), required=True)
    p.add_argument("--uri", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--project", required=True)
    p.add_argument("--gto-secret")
    args = p.parse_args(argv)
    if args.role == "train":
        try:
            run_experiment(args.output, upload_uri=args.uri)
        finally:
            if args.output.exists():
                # Retain latest full state, all policy exports, metrics and journals.
                try:
                    if (args.output / "training" / "latest.json").exists():
                        upload_recovery(args.output, args.uri)
                finally:
                    # Recovery pointers and endpoint markers have a transactional
                    # publication order. An unordered rsync must never bypass it.
                    sync_output(args.output, args.uri, exclude=(
                        r"^training/(?:state_[0-9]+(?:/.*)?|latest\.json|manifest\.json)$"
                        r"|^P[48]\.(?:json|pkl)$|^cloud_checkpoint\.json$"))
    else:
        external_worker(args.output, args.uri, secret=args.gto_secret, project=args.project)


if __name__ == "__main__":
    main()
