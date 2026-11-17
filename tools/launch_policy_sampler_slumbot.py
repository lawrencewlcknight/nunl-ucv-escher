#!/usr/bin/env python3
"""Explicitly launch the small Slumbot VM only after the local sampler test passes."""
import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess


def read_remote(uri):
    result = subprocess.run(["gcloud", "storage", "cat", uri], check=True,
                            capture_output=True, text=True, timeout=60)
    return json.loads(result.stdout)


def build_bundle(plan, ready, output):
    if plan.get("experiment") != "hunl_policy_sampler_v1" or ready.get("status") != "READY":
        raise ValueError("A completed positive local sampler test is required")
    if ready.get("external_fit_seed") != plan["fit_seeds"][0]:
        raise ValueError("Follow-up fit seed differs from the predeclared seed")
    digest, experiment_hash = plan["source_sha256"], ready["experiment_sha256"]
    if not all(re.fullmatch(r"[a-f0-9]{64}", x) for x in (digest, experiment_hash)):
        raise ValueError("Invalid pinned source or experiment hash")
    command = next(c for c in plan["commands"] if c[1:4] == ["compute", "instances", "create"])
    option = lambda name: command[command.index(name) + 1]
    name = command[4] + "-slumbot"
    if not re.fullmatch(r"[a-z][a-z0-9-]{0,61}[a-z0-9]", name):
        raise ValueError("Invalid Slumbot VM name")
    uri = plan["uri"].rstrip("/")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    worker = ["/opt/hunl/venv/bin/python", "-u", "-m", "hunl_escher.policy_sampling_cloud",
        "--role", "slumbot", "--workspace", "/var/lib/hunl-sampler-slumbot",
        "--source-uri", uri, "--output-uri", uri + "/slumbot",
        "--expected-experiment-sha256", experiment_hash]
    script = output / "startup.sh"
    script.write_text("\n".join([
        "#!/bin/bash", "set -euo pipefail", "export DEBIAN_FRONTEND=noninteractive",
        "if [ -e /var/lib/hunl-sampler-slumbot-started ]; then shutdown -h now; exit 0; fi",
        "touch /var/lib/hunl-sampler-slumbot-started",
        "finish() { rc=$?; trap - EXIT; set +e; "
        + shlex.join(["timeout", "60", "gcloud", "storage", "cp", "/var/log/hunl-sampler-slumbot.log", uri + "/slumbot/startup.log"])
        + '; shutdown -h now; exit "$rc"; }', "trap finish EXIT",
        "exec > >(tee -a /var/log/hunl-sampler-slumbot.log) 2>&1",
        "apt-get update -qq", "apt-get install -y -qq python3-venv google-cloud-cli",
        "mkdir -p /opt/hunl/source", "cd /opt/hunl/source",
        shlex.join(["gcloud", "storage", "cp", uri + "/source.tar.gz", "/opt/hunl/source.tar.gz"]),
        f"echo '{digest}  /opt/hunl/source.tar.gz' | sha256sum --check",
        "tar -xzf /opt/hunl/source.tar.gz -C /opt/hunl/source",
        "python3 -m venv /opt/hunl/venv",
        "/opt/hunl/venv/bin/python -m pip install --upgrade pip",
        "/opt/hunl/venv/bin/python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu",
        "/opt/hunl/venv/bin/python -m pip install '.[api]'",
        "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1",
        shlex.join(worker), ""]))
    launch = dict(uri=uri + "/slumbot", expected_experiment_sha256=experiment_hash,
        source_sha256=digest, benchmark_max_hours=24, vm_max_hours=26,
        commands=[["gcloud", "storage", "cp", "--if-generation-match=0", str(output / "launch.json"), uri + "/slumbot/launch.json"],
        ["gcloud", "compute", "instances", "create", name, "--project", option("--project"), "--zone", option("--zone"),
         "--machine-type", "c4-standard-4", "--image-project", "debian-cloud", "--image-family", "debian-12",
         "--boot-disk-type", "hyperdisk-balanced", "--boot-disk-size", "30GB",
         "--service-account", option("--service-account"), "--scopes", "cloud-platform",
         "--metadata-from-file", "startup-script=" + str(script), "--no-restart-on-failure",
         "--max-run-duration", "26h", "--instance-termination-action", "STOP",
         "--labels", "purpose=hunl-sampler-slumbot"]])
    (output / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
    return launch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--local-plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    plan = json.loads(args.local_plan.read_text())
    if plan.get("experiment") != "hunl_policy_sampler_v1":
        raise ValueError("Expected the local sampler launch plan")
    ready = read_remote(plan["uri"].rstrip("/") + "/slumbot_ready.json")
    launch = build_bundle(plan, ready, args.output)
    print(json.dumps(launch, indent=2))
    if args.execute:
        for command in launch["commands"]:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
