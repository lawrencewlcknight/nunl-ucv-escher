"""Inference-only Batch worker for independent HUNL P128 external benchmarks."""
import argparse
import json
import os
from pathlib import Path
import signal

from .checkpointing import sha256_file
from .gtow_p32_cloud import publication, read_secret, storage
from .p128_benchmark import run_benchmark, verified_candidate, POLICY_SHA256, MANIFEST_SHA256


def download_policy(source_uri, root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    for name, expected in (("continuation.json", MANIFEST_SHA256), ("P128.json", None),
                           ("P128.pkl", POLICY_SHA256)):
        target = root / name
        storage("cp", source_uri.rstrip("/") + "/" + name, target)
        if expected is not None and sha256_file(target) != expected:
            raise ValueError("P128 artifact checksum mismatch: " + name)
    verified_candidate(root)
    return root


def run_worker(workspace, source_uri, output_uri, provider, project, gto_secret):
    source_uri, output_uri = source_uri.rstrip("/"), output_uri.rstrip("/")
    if (source_uri == output_uri or source_uri.startswith(output_uri + "/")
            or output_uri.startswith(source_uri + "/")):
        raise ValueError("Use a separate benchmark output prefix")
    with publication(workspace, output_uri) as (output, status, stage):
        root = download_policy(source_uri, Path(workspace) / "policy")
        status.update(source_uri=source_uri, policy_sha256=POLICY_SHA256,
                      source_manifest_sha256=MANIFEST_SHA256, provider=provider)
        previous = os.environ.get("GTO_WIZARD_API_KEY")
        try:
            if provider == "gtow":
                os.environ["GTO_WIZARD_API_KEY"] = read_secret(project, gto_secret)
            # run_benchmark already verifies the policy and authenticated GETs
            # before its first hand; a separate preflight repeats the same work.
            stage("BENCHMARK")
            report = run_benchmark(root, output / "benchmark", provider=provider, max_seconds=86400)
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
    parser.add_argument("--provider", required=True, choices=("slumbot", "gtow"))
    parser.add_argument("--project", required=True)
    parser.add_argument("--gto-secret", default="gtowizard-benchmark-api-key")
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
