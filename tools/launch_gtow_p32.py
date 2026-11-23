#!/usr/bin/env python3
"""Prepare a Google Cloud Batch benchmark job; --execute uploads and submits it."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess

from launch_learning_check import source_archive

SOURCE_URI = "gs://clever-overview-399515-fhp-escher-results/hunl/hunl-sampler-p32-20261117-170440"
REFIT_SHA256 = "8df5ecc1e088f38e6dd7f1918e71b30d1869d76b17144f61b6642101281590b5"
POLICY_SHA256 = "add9bf970e9979be6b74c0548b084d1af19e1c8b334bd4ef326f192ceafabe29"
# Batch can omit PATH; Bash's fallback need not be exported to Python children.
BATCH_PATH = 'export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin${PATH:+:$PATH}"'


def build_bundle(output, *, run_name, project="clever-overview-399515", region="europe-west1",
                 bucket="gs://clever-overview-399515-fhp-escher-results/hunl",
                 service_account="fhp-escher-runner@clever-overview-399515.iam.gserviceaccount.com",
                 gto_secret="gtowizard-benchmark-api-key", source_uri=SOURCE_URI,
                 expected_experiment_sha256=REFIT_SHA256, expected_policy_sha256=POLICY_SHA256):
    if not re.fullmatch(r"[a-z][a-z0-9-]{0,61}[a-z0-9]", run_name):
        raise ValueError("Use a lowercase Batch job name of 2–63 letters, digits and hyphens")
    if not re.fullmatch(r"[a-z]+-[a-z]+[0-9]+", region):
        raise ValueError("Use a Google Cloud region such as europe-west1")
    for uri in (bucket, source_uri):
        if not re.fullmatch(r"gs://[a-z0-9][a-z0-9._-]+(?:/[a-zA-Z0-9_./-]+)?", uri):
            raise ValueError("Invalid Cloud Storage URI")
    if not all(re.fullmatch(r"[a-f0-9]{64}", v) for v in (expected_experiment_sha256, expected_policy_sha256)):
        raise ValueError("Expected artifact SHA-256 values are required")
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", gto_secret):
        raise ValueError("Use a Secret Manager secret name, not the API key")
    uri = bucket.rstrip("/") + "/" + run_name
    source_uri = source_uri.rstrip("/")
    if uri == source_uri or uri.startswith(source_uri + "/") or source_uri.startswith(uri + "/"):
        raise ValueError("Use a separate output prefix")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_archive(Path(__file__).resolve().parents[1], output / "source.tar.gz")
    digest = hashlib.sha256((output / "source.tar.gz").read_bytes()).hexdigest()
    worker = ["/opt/hunl/venv/bin/python", "-I", "-u", "-m", "hunl_escher.gtow_p32_cloud",
        "--workspace", "/var/lib/hunl-gtow-p32", "--source-uri", source_uri,
        "--output-uri", uri, "--expected-experiment-sha256", expected_experiment_sha256,
        "--expected-policy-sha256", expected_policy_sha256,
        "--project", project, "--gto-secret", gto_secret]
    script = output / "run.sh"
    script.write_text("\n".join([
        "#!/bin/bash", "set -Eeuo pipefail", BATCH_PATH, "umask 077", "export DEBIAN_FRONTEND=noninteractive",
        "exec > >(tee -a /var/log/hunl-gtow-p32.log) 2>&1",
        "GTO_SETUP_STAGE=initialization",
        'stage() { GTO_SETUP_STAGE="$1"; printf "GTO stage=%s\\n" "$GTO_SETUP_STAGE"; }',
        "trap 'printf \"ERROR: stage=%s line=%s exit_code=%s\\n\" \"$GTO_SETUP_STAGE\" \"$LINENO\" \"$?\" >&2' ERR",
        "stage system_packages",
        "apt-get -o Acquire::Retries=3 update -qq",
        # Batch images may still be running unattended-upgrades when tasks start.
        # Wait using apt's own lock mechanism; never delete locks or kill dpkg.
        "apt-get -o DPkg::Lock::Timeout=600 -o Acquire::Retries=3 install -y -qq --no-upgrade python3-venv google-cloud-cli",
        "stage source_download",
        "mkdir -p /opt/hunl/source", "cd /opt/hunl/source",
        shlex.join(["gcloud", "storage", "cp", uri + "/source.tar.gz", "/opt/hunl/source.tar.gz"]),
        f"echo '{digest}  /opt/hunl/source.tar.gz' | sha256sum --check",
        "tar -xzf /opt/hunl/source.tar.gz -C /opt/hunl/source",
        "stage python_diagnostics",
        shlex.join(["/usr/bin/python3", "-I", "-c",
            "import json, os, sys; print(json.dumps(dict(version=sys.version, executable=sys.executable, "
            "base_executable=sys._base_executable))); assert sys._base_executable and "
            "os.access(sys._base_executable, os.X_OK), 'Base Python executable is missing or not executable'"]),
        "stage python_venv",
        "/usr/bin/python3 -I -m venv /opt/hunl/venv",
        "stage dependencies",
        "/opt/hunl/venv/bin/python -I -m pip install --upgrade pip",
        "/opt/hunl/venv/bin/python -I -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu",
        "/opt/hunl/venv/bin/python -I -m pip install '.[api]'",
        "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1",
        "stage benchmark_worker",
        shlex.join(["timeout", "--signal=TERM", "--kill-after=600s", "25h", *worker]), ""]))
    # This runs after a failed main runnable too. Task expiry/host loss can still
    # prevent cleanup, so the worker also publishes immutable snapshots regularly.
    cleanup = output / "publish.sh"
    cleanup.write_text("\n".join([
        "#!/bin/bash", "set -euo pipefail", BATCH_PATH, "umask 077", "rc=0",
        "if [ -d /var/lib/hunl-gtow-p32/results ]; then",
        "  " + shlex.join(["timeout", "300", "gcloud", "storage", "rsync", "--recursive",
            "--exclude", r"(^|/)[^/]*\.tmp$|^cloud_status\.json$",
            "/var/lib/hunl-gtow-p32/results", uri]) + " || rc=1",
        "  if [ \"$rc\" -eq 0 ] && [ -f /var/lib/hunl-gtow-p32/results/cloud_status.json ]; then",
        "    " + shlex.join(["timeout", "120", "gcloud", "storage", "cp",
            "/var/lib/hunl-gtow-p32/results/cloud_status.json", uri + "/cloud_status.json"]) + " || rc=1",
        "  fi", "fi",
        "if [ -f /var/log/hunl-gtow-p32.log ]; then",
        "  " + shlex.join(["timeout", "120", "gcloud", "storage", "cp",
            "/var/log/hunl-gtow-p32.log", uri + "/startup.log"]) + " || rc=1",
        "fi", 'exit "$rc"', ""]))
    job = dict(
        taskGroups=[dict(taskCount=1, parallelism=1, taskSpec=dict(
            runnables=[dict(script=dict(text=script.read_text()), timeout="91800s"),
                       dict(script=dict(text=cleanup.read_text()), alwaysRun=True, timeout="900s")],
            computeResource=dict(cpuMilli=2000, memoryMib=6144),
            maxRetryCount=0, maxRunDuration="93600s"))],
        allocationPolicy=dict(
            location=dict(allowedLocations=["regions/" + region]),
            serviceAccount=dict(email=service_account, scopes=["https://www.googleapis.com/auth/cloud-platform"]),
            instances=[dict(policy=dict(machineType="c4-standard-2", provisioningModel="STANDARD",
                bootDisk=dict(sizeGb=30, type="hyperdisk-balanced",
                    image="projects/batch-custom-image/global/images/family/batch-debian-12-official")))]),
        logsPolicy=dict(destination="CLOUD_LOGGING"), labels=dict(purpose="hunl-gtow-p32"))
    (output / "job.json").write_text(json.dumps(job, indent=2) + "\n")
    commands = [
        ["gcloud", "storage", "cp", "--if-generation-match=0", str(output / "launch.json"), uri + "/launch.json"],
        ["gcloud", "storage", "cp", "--if-generation-match=0", str(output / "source.tar.gz"), uri + "/source.tar.gz"],
        ["gcloud", "storage", "cp", "--if-generation-match=0", str(output / "job.json"), uri + "/job.json"],
        ["gcloud", "batch", "jobs", "submit", run_name, "--project", project, "--location", region,
         "--config", str(output / "job.json")]]
    plan = dict(experiment="gtow_corrected_p32_v1", backend="google_cloud_batch", uri=uri, source_uri=source_uri,
        expected_experiment_sha256=expected_experiment_sha256, expected_policy_sha256=expected_policy_sha256,
        source_sha256=digest, machine="c4-standard-2", disk_gb=30, threads=1,
        project=project, region=region, job_name=run_name, gto_secret=gto_secret,
        pilot_hands=1000, confirmation_hands=20000, benchmark_max_hours=24, task_max_hours=26,
        max_retry_count=0, publication_interval_seconds=120,
        cleanup="Batch deletes worker resources; reports and journal snapshots persist in Cloud Storage.",
        commands=commands)
    (output / "launch.json").write_text(json.dumps(plan, indent=2) + "\n")
    return plan


def verify_remote_manifest(plan):
    result = subprocess.run(["gcloud", "storage", "cat", plan["source_uri"] + "/fit/experiment.json"],
                            check=True, capture_output=True, timeout=60)
    if hashlib.sha256(result.stdout).hexdigest() != plan["expected_experiment_sha256"]:
        raise ValueError("Remote refit manifest differs from the pinned experiment; no Batch job submitted")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--project", default="clever-overview-399515")
    parser.add_argument("--region", default="europe-west1")
    parser.add_argument("--bucket", default="gs://clever-overview-399515-fhp-escher-results/hunl")
    parser.add_argument("--service-account", default="fhp-escher-runner@clever-overview-399515.iam.gserviceaccount.com")
    parser.add_argument("--gto-secret", default="gtowizard-benchmark-api-key")
    parser.add_argument("--execute", action="store_true")
    args = vars(parser.parse_args(argv))
    execute = args.pop("execute")
    plan = build_bundle(**args)
    print(json.dumps(plan, indent=2))
    if execute:
        verify_remote_manifest(plan)
        # The first generation-guarded write reserves this run. Do not blindly
        # repeat a failed submission: inspect its launch record and Batch job first.
        for command in plan["commands"]:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
