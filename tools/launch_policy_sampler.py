#!/usr/bin/env python3
"""Prepare a bounded CPU VM launch; --execute uploads and starts the experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess

from launch_learning_check import source_archive


def build_bundle(output, *, project, zone, bucket, service_account, run_name, source_uri,
                 expected_policy_sha256, iteration=1658):
    if not re.fullmatch(r"[a-z][a-z0-9-]{0,45}[a-z0-9]", run_name):
        raise ValueError("Invalid run name")
    for uri in (bucket, source_uri):
        if not re.fullmatch(r"gs://[a-z0-9][a-z0-9._-]+(?:/[a-zA-Z0-9_./-]+)?", uri):
            raise ValueError("Invalid Cloud Storage URI")
    if not re.fullmatch(r"[a-f0-9]{64}", expected_policy_sha256) or iteration < 1:
        raise ValueError("Expected policy SHA-256 and positive iteration are required")
    uri, source_uri = bucket.rstrip("/") + "/" + run_name, source_uri.rstrip("/")
    if uri == source_uri or uri.startswith(source_uri + "/") or source_uri.startswith(uri + "/"):
        raise ValueError("Use a separate cloud output prefix")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_archive(Path(__file__).resolve().parents[1], output / "source.tar.gz")
    digest = hashlib.sha256((output / "source.tar.gz").read_bytes()).hexdigest()
    command = ["/opt/hunl/venv/bin/python", "-u", "-m", "hunl_escher.policy_sampling_cloud",
        "--workspace", "/var/lib/hunl-policy-sampler", "--source-uri", source_uri,
        "--output-uri", uri, "--expected-policy-sha256", expected_policy_sha256,
        "--iteration", str(iteration)]
    script = output / "startup.sh"
    script.write_text("\n".join([
        "#!/bin/bash", "set -euo pipefail", "export DEBIAN_FRONTEND=noninteractive",
        "if [ -e /var/lib/hunl-policy-sampler-started ]; then shutdown -h now; exit 0; fi",
        "touch /var/lib/hunl-policy-sampler-started",
        "finish() { rc=$?; trap - EXIT; set +e; "
        + shlex.join(["timeout", "60", "gcloud", "storage", "cp", "/var/log/hunl-policy-sampler.log", uri + "/startup.log"])
        + '; shutdown -h now; exit "$rc"; }', "trap finish EXIT",
        "exec > >(tee -a /var/log/hunl-policy-sampler.log) 2>&1",
        "apt-get update -qq", "apt-get install -y -qq python3-venv google-cloud-cli ",
        "mkdir -p /opt/hunl/source", "cd /opt/hunl/source",
        shlex.join(["gcloud", "storage", "cp", uri + "/source.tar.gz", "/opt/hunl/source.tar.gz"]),
        f"echo '{digest}  /opt/hunl/source.tar.gz' | sha256sum --check",
        "tar -xzf /opt/hunl/source.tar.gz -C /opt/hunl/source",
        "python3 -m venv /opt/hunl/venv",
        "/opt/hunl/venv/bin/python -m pip install --upgrade pip",
        "/opt/hunl/venv/bin/python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu",
        "/opt/hunl/venv/bin/python -m pip install '.[api]'",
        "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1",
        shlex.join(command), ""]))
    commands = [
        ["gcloud", "storage", "cp", "--if-generation-match=0", str(output / "source.tar.gz"), uri + "/source.tar.gz"],
        ["gcloud", "storage", "cp", "--if-generation-match=0", str(output / "launch.json"), uri + "/launch.json"],
        ["gcloud", "compute", "instances", "create", run_name, "--project", project, "--zone", zone,
         "--machine-type", "c4-standard-16", "--image-project", "debian-cloud", "--image-family", "debian-12",
         "--boot-disk-type", "hyperdisk-balanced", "--boot-disk-size", "100GB",
         "--service-account", service_account, "--scopes", "cloud-platform",
         "--metadata-from-file", "startup-script=" + str(script), "--no-restart-on-failure",
         "--max-run-duration", "6h", "--instance-termination-action", "STOP",
         "--labels", "purpose=hunl-policy-sampler"]]
    plan = dict(experiment="hunl_policy_sampler_v1", uri=uri, source_uri=source_uri, expected_policy_sha256=expected_policy_sha256,
        iteration=iteration, source_sha256=digest, machine="c4-standard-16", threads=8,
        evaluation_workers=8, fit_seeds=[1701, 1702], evaluation_max_seconds=7200,
        vm_max_hours=6, commands=commands,
        shutdown="Automatic on exit; six-hour VM safety cap. Stopped boot disk remains billable.",
        data="Only policy replay (about 18.7 GB); no self-play collection or full solver restore")
    (output / "launch.json").write_text(json.dumps(plan, indent=2) + "\n")
    return plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", required=True)
    p.add_argument("--zone", default="europe-west1-b")
    p.add_argument("--bucket", required=True)
    p.add_argument("--service-account", required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--source-uri", required=True)
    p.add_argument("--expected-policy-sha256", required=True)
    p.add_argument("--iteration", type=int, default=1658)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--execute", action="store_true")
    args = vars(p.parse_args())
    execute = args.pop("execute")
    plan = build_bundle(**args)
    print(json.dumps(plan, indent=2))
    if execute:
        # Generation preconditions reject existing source/launch objects before VM creation.
        for command in plan["commands"]:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
