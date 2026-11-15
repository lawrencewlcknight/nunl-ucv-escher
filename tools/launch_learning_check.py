#!/usr/bin/env python3
"""Prepare a reviewable GCE launch bundle; --execute explicitly submits paid VMs."""
import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess
import tarfile


def source_archive(repository, destination):
    # Explicit source allowlist: never upload .env, credentials, .git or outputs.
    packages = ("hunl_escher", "unbiased_escher", "adaptive_escher", "vr_deep_cfr")
    with tarfile.open(destination, "w:gz") as archive:
        for name in ("pyproject.toml", "README.md"):
            archive.add(repository / name, arcname=name, recursive=False)
        for package in packages:
            for path in sorted((repository / package).rglob("*")):
                if path.is_file() and not path.is_symlink() and path.suffix in (".py", ".json"):
                    archive.add(path, arcname=str(path.relative_to(repository)), recursive=False)


def startup(role, uri, project, secret):
    command = ["/opt/hunl/venv/bin/python", "-m", "hunl_escher.cloud_worker", "--role", role,
               "--uri", uri, "--project", project, "--output", "/var/lib/hunl-run"]
    if role == "external" and secret:
        command += ["--gto-secret", secret]
    return "\n".join([
        "#!/bin/bash", "set -euo pipefail", "export DEBIAN_FRONTEND=noninteractive",
        "# A manual restart must never silently launch a second paid experiment.",
        "if [ -e /var/lib/hunl-launch-started ]; then shutdown -h now; exit 0; fi",
        "touch /var/lib/hunl-launch-started", "trap 'shutdown -h now' EXIT",
        "apt-get update -qq", "apt-get install -y -qq python3-venv google-cloud-cli",
        "mkdir -p /opt/hunl/source", "cd /opt/hunl/source",
        shlex.join(["gcloud", "storage", "cp", uri+"/source.tar.gz", "/opt/hunl/source.tar.gz"]),
        "tar -xzf /opt/hunl/source.tar.gz -C /opt/hunl/source",
        "python3 -m venv /opt/hunl/venv",
        "/opt/hunl/venv/bin/python -m pip install --upgrade pip",
        "/opt/hunl/venv/bin/python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu",
        "/opt/hunl/venv/bin/python -m pip install '.[parallel,api]'",
        "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1",
        shlex.join(command), "",
    ])


def build_bundle(output, *, project, zone, bucket, service_account, run_name, gto_secret):
    if not re.fullmatch(r"[a-z][a-z0-9-]{0,45}[a-z0-9]", run_name):
        raise ValueError("Use a lowercase run name, 2–47 letters/digits/hyphens")
    if not re.fullmatch(r"gs://[a-z0-9][a-z0-9._-]+(?:/[a-zA-Z0-9_./-]+)?", bucket):
        raise ValueError("Use a gs:// bucket/prefix without shell characters")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    uri = bucket.rstrip("/")+"/"+run_name
    repository = Path(__file__).resolve().parents[1]
    source_archive(repository, output / "source.tar.gz")
    commands = [["gcloud", "storage", "cp", "--if-generation-match=0", str(output / "source.tar.gz"), uri+"/source.tar.gz"]]
    for role, machine, disk, hours in (("train", "c4-highmem-16", "200GB", 15),
                                        ("external", "c4-standard-4", "30GB", 26)):
        script = output / f"{role}-startup.sh"
        script.write_text(startup(role, uri, project, gto_secret))
        commands.append(["gcloud", "compute", "instances", "create", run_name+"-"+role,
            "--project", project, "--zone", zone, "--machine-type", machine,
            "--image-project", "debian-cloud", "--image-family", "debian-12",
            "--boot-disk-type", "hyperdisk-balanced", "--boot-disk-size", disk,
            "--service-account", service_account, "--scopes", "cloud-platform",
            "--metadata-from-file", "startup-script="+str(script),
            "--no-restart-on-failure", "--max-run-duration", f"{hours}h",
            "--instance-termination-action", "STOP", "--labels", "purpose=hunl-learning-check"])
    plan = dict(uri=uri, commands=commands,
                training_active_hours=8, training_wall_budget_hours=12, local_evaluation_hours=2,
                external_budget_hours=24, vm_limits_hours={"train": 15, "external": 26},
                estimated_compute_ceiling_usd=15*1.042712+26*.19767,
                price_note="Planning rates for us-central1, not a billing quote; disks/network/provider fees extra",
                cleanup="VMs stop automatically; delete stopped VMs/disks after verifying artifacts to end storage charges")
    (output / "launch.json").write_text(json.dumps(plan, indent=2)+"\n")
    return plan


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", required=True)
    p.add_argument("--zone", default="us-central1-a")
    p.add_argument("--bucket", required=True, help="Existing private gs:// bucket/prefix")
    p.add_argument("--service-account", required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--gto-secret", default="gtowizard-benchmark-api-key")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--execute", action="store_true")
    args = vars(p.parse_args(argv))
    execute = args.pop("execute")
    plan = build_bundle(**args)
    print(json.dumps(plan, indent=2))
    if execute:
        # Ensure existing remote artifacts cannot be mistaken for this run's P4/P8.
        existing = subprocess.run(["gcloud", "storage", "ls", plan["uri"]+"/**"], capture_output=True, text=True)
        if existing.returncode == 0:
            raise RuntimeError("Remote run prefix already exists; choose a new run name")
        created = []
        try:
            for command in plan["commands"]:
                subprocess.run(command, check=True)
                if command[1:4] == ["compute", "instances", "create"]:
                    created.append(command)
        except subprocess.CalledProcessError:
            # Do not leave training running if provisioning the evaluator fails.
            # Stop only VMs successfully created by this invocation; retain disks.
            for command in created:
                subprocess.run(["gcloud", "compute", "instances", "stop", command[4],
                    "--project", command[command.index("--project")+1],
                    "--zone", command[command.index("--zone")+1], "--quiet"], check=False)
            raise


if __name__ == "__main__":
    main()
