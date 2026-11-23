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

SOURCE_URI = "gs://clever-overview-399515-fhp-escher-results/hunl/hunl-p104-plus24-batch-20261122"
MANIFEST_SHA256 = "a08210f79bcddebc564109886673d38cb7c8002326b682b2b66f085eb54c048b"
POLICY_SHA256 = "aade5f09a22fc7aeddc7063e4250fc00f04c16cc5c4827381f775e3d66727bb3"
# Batch can omit PATH; Bash's fallback need not be exported to Python children.
BATCH_PATH = 'export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin${PATH:+:$PATH}"'


def build_bundle(output, *, provider, run_name, project="clever-overview-399515", region="europe-west1",
                 bucket="gs://clever-overview-399515-fhp-escher-results/hunl",
                 service_account="fhp-escher-runner@clever-overview-399515.iam.gserviceaccount.com",
                 gto_secret="gtowizard-benchmark-api-key"):
    if provider not in ("slumbot", "gtow"):
        raise ValueError("Unknown benchmark provider")
    source_uri = SOURCE_URI
    expected_experiment_sha256, expected_policy_sha256 = MANIFEST_SHA256, POLICY_SHA256
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
    worker = ["/opt/hunl/venv/bin/python", "-I", "-u", "-m", "hunl_escher.p128_benchmark_cloud",
        "--workspace", "/var/lib/hunl-p128-external", "--source-uri", source_uri,
        "--output-uri", uri, "--provider", provider,
        "--project", project, "--gto-secret", gto_secret]
    script = output / "run.sh"
    script.write_text("\n".join([
        "#!/bin/bash", "set -Eeuo pipefail", BATCH_PATH, "umask 077", "export DEBIAN_FRONTEND=noninteractive",
        "exec > >(tee -a /var/log/hunl-p128-external.log) 2>&1",
        "BENCHMARK_SETUP_STAGE=initialization",
        'stage() { BENCHMARK_SETUP_STAGE="$1"; printf "Benchmark stage=%s\\n" "$BENCHMARK_SETUP_STAGE"; }',
        "trap 'printf \"ERROR: stage=%s line=%s exit_code=%s\\n\" \"$BENCHMARK_SETUP_STAGE\" \"$LINENO\" \"$?\" >&2' ERR",
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
        "if [ -d /var/lib/hunl-p128-external/results ]; then",
        "  " + shlex.join(["timeout", "300", "gcloud", "storage", "rsync", "--recursive",
            "--exclude", r"(^|/)[^/]*\.tmp$|^cloud_status\.json$",
            "/var/lib/hunl-p128-external/results", uri]) + " || rc=1",
        "  if [ \"$rc\" -eq 0 ] && [ -f /var/lib/hunl-p128-external/results/cloud_status.json ]; then",
        "    " + shlex.join(["timeout", "120", "gcloud", "storage", "cp",
            "/var/lib/hunl-p128-external/results/cloud_status.json", uri + "/cloud_status.json"]) + " || rc=1",
        "  fi", "fi",
        "if [ -f /var/log/hunl-p128-external.log ]; then",
        "  " + shlex.join(["timeout", "120", "gcloud", "storage", "cp",
            "/var/log/hunl-p128-external.log", uri + "/startup.log"]) + " || rc=1",
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
        logsPolicy=dict(destination="CLOUD_LOGGING"), labels=dict(purpose="hunl-p128-" + provider))
    (output / "job.json").write_text(json.dumps(job, indent=2) + "\n")
    commands = [
        ["gcloud", "storage", "cp", "--if-generation-match=0", str(output / "launch.json"), uri + "/launch.json"],
        ["gcloud", "storage", "cp", "--if-generation-match=0", str(output / "source.tar.gz"), uri + "/source.tar.gz"],
        ["gcloud", "storage", "cp", "--if-generation-match=0", str(output / "job.json"), uri + "/job.json"],
        ["gcloud", "batch", "jobs", "submit", run_name, "--project", project, "--location", region,
         "--config", str(output / "job.json")]]
    plan = dict(experiment="hunl_p128_external_v1", provider=provider, backend="google_cloud_batch", uri=uri, source_uri=source_uri,
        expected_manifest_sha256=expected_experiment_sha256, expected_policy_sha256=expected_policy_sha256,
        source_sha256=digest, job_sha256=hashlib.sha256((output / "job.json").read_bytes()).hexdigest(), machine="c4-standard-2", disk_gb=30, threads=1,
        project=project, region=region, job_name=run_name, gto_secret=gto_secret,
        pilot_hands=0,
        confirmation_hands=20000 if provider == "slumbot" else 8000,
        total_hand_cap=20000 if provider == "slumbot" else 8000, request_timeout_seconds=180, benchmark_max_hours=24, task_max_hours=26,
        max_retry_count=0, publication_interval_seconds=120,
        cleanup="Batch deletes worker resources; reports and journal snapshots persist in Cloud Storage.",
        commands=commands)
    (output / "launch.json").write_text(json.dumps(plan, indent=2) + "\n")
    return plan


def verify_remote_manifest(plan):
    result = subprocess.run(["gcloud", "storage", "cat", plan["source_uri"] + "/continuation.json"],
                            check=True, capture_output=True, timeout=60)
    if hashlib.sha256(result.stdout).hexdigest() != plan["expected_manifest_sha256"]:
        raise ValueError("Remote training manifest differs from the pinned P128 run; no Batch job submitted")


def submit_prepared(directory):
    directory = Path(directory).resolve()
    plan = json.loads((directory / "launch.json").read_text())
    if (plan["experiment"] != "hunl_p128_external_v1" or plan["source_uri"] != SOURCE_URI
            or plan["expected_policy_sha256"] != POLICY_SHA256
            or plan["expected_manifest_sha256"] != MANIFEST_SHA256):
        raise ValueError("Prepared plan does not describe the pinned P128 experiment")
    for name, key in (("source.tar.gz", "source_sha256"), ("job.json", "job_sha256")):
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != plan[key]:
            raise ValueError("Prepared bundle changed: " + name)
    verify_remote_manifest(plan)
    # Reserve the output prefix before upload/submission. A second invocation
    # fails rather than replacing reports or replaying the quota-consuming job.
    commands = [["gcloud", "storage", "cp", "--if-generation-match=0",
                 str(directory / name), plan["uri"] + "/" + name]
                for name in ("launch.json", "source.tar.gz", "job.json")]
    commands.append(["gcloud", "batch", "jobs", "submit", plan["job_name"],
        "--project", plan["project"], "--location", plan["region"], "--config", str(directory / "job.json")])
    for command in commands:
        subprocess.run(command, check=True)
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("slumbot", "gtow"))
    parser.add_argument("--run-name")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--submit-prepared", type=Path, help="Submit an existing checked launch bundle")
    parser.add_argument("--project", default="clever-overview-399515")
    parser.add_argument("--region", default="europe-west1")
    parser.add_argument("--bucket", default="gs://clever-overview-399515-fhp-escher-results/hunl")
    parser.add_argument("--service-account", default="fhp-escher-runner@clever-overview-399515.iam.gserviceaccount.com")
    parser.add_argument("--gto-secret", default="gtowizard-benchmark-api-key")
    parser.add_argument("--execute", action="store_true")
    args = vars(parser.parse_args(argv))
    prepared, execute = args.pop("submit_prepared"), args.pop("execute")
    if prepared:
        if execute or args["provider"] or args["run_name"] or args["output"]:
            parser.error("--submit-prepared cannot be combined with build options or --execute")
        plan = submit_prepared(prepared)
    else:
        if not all(args[k] for k in ("provider", "run_name", "output")):
            parser.error("Preparing a bundle requires --provider, --run-name and --output")
        plan = build_bundle(**args)
        if execute:
            submit_prepared(args["output"])
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
