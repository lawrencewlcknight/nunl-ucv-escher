"""Shared, pinned Google Batch runtime for No-Limit Hold’em experiments."""
import shlex

# Debian 12's supported Python is 3.11.2, matching the original VM checkpoint.
# The old python:3.11.2 image used now-retired Debian 11 package repositories.
PYTHON_IMAGE = "docker.io/library/debian@sha256:c67e92c8099ba22853ab4cfda7cfb494d253d6012ca82dee98ba8976772af018"
VM_IMAGE = "projects/batch-custom-image/global/images/batch-cos-stable-official-20261110-00-p01"


def bootstrap(command, uri, digest):
    """Install the checkpoint-compatible runtime and preserve exit logs."""
    script = "\n".join([
        "set -Eeuo pipefail",
        "export DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1",
        "export CLOUDSDK_PYTHON=/usr/bin/python3",
        "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1",
        "mkdir -p /workspace/coordinator",
        "exec > >(tee -a /workspace/batch-run.log) 2>&1",
        "finish() {",
        '  code=$?; trap - EXIT; set +e',
        '  printf \'{"exit_code":%s}\\n\' "$code" > /workspace/batch-exit.json',
        "  if command -v gcloud >/dev/null; then",
        "    for attempt in 1 2 3 4; do",
        "      gcloud storage cp /workspace/batch-run.log /workspace/batch-exit.json " + shlex.quote(uri + "/") + ' && break',
        '      if [ "$attempt" = 4 ] && [ "$code" = 0 ]; then code=1; fi',
        "      sleep 5",
        "    done",
        "  fi",
        '  exit "$code"',
        "}",
        "trap finish EXIT",
        "trap 'exit 143' TERM",
        "apt-get -o APT::Update::Error-Mode=any -o Acquire::Retries=3 update -qq",
        "apt-get -o Acquire::Retries=3 install -y -qq ca-certificates curl gnupg python3-venv procps",
        "curl --fail --retry 3 https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg",
        "echo 'deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main' > /etc/apt/sources.list.d/google-cloud-sdk.list",
        "apt-get -o APT::Update::Error-Mode=any -o Acquire::Retries=3 update -qq",
        "apt-get install -y -qq google-cloud-cli",
        "python3 -m venv /workspace/venv",
        'export PATH="/workspace/venv/bin:$PATH"',
        "python -c 'import platform; assert platform.python_version() == \"3.11.2\", platform.python_version()'",
        shlex.join(["gcloud", "storage", "cp", uri + "/source.tar.gz", "/workspace/source.tar.gz"]),
        f"echo '{digest}  /workspace/source.tar.gz' | sha256sum --check",
        "tar -xzf /workspace/source.tar.gz -C /workspace/coordinator",
        "cd /workspace/coordinator",
        "python -m pip install --no-cache-dir pip==25.3 setuptools==80.9.0 wheel==0.45.1",
        "python -m pip install --no-cache-dir --no-deps --only-binary=:all: torch==2.7.0+cpu --index-url https://download.pytorch.org/whl/cpu",
        "python -m pip install --no-cache-dir '.[parallel]'",
        "python -m pip check",
        "python --version",
        "uname -r || true",
        "free -m || true",
        "df -h /workspace || true",
        shlex.join(command), "",
    ])
    return script


def job_spec(script, service_account, *, boot_disk_gb, max_run_seconds, labels):
    """One standard CPU task, no automatic rerun of expensive experiments."""
    job = dict(taskGroups=[dict(taskCount=1, parallelism=1, taskCountPerNode=1,
        taskSpec=dict(runnables=[dict(container=dict(imageUri=PYTHON_IMAGE, entrypoint="/bin/bash",
            commands=["-c", script], options="--shm-size=1g"))],
            computeResource=dict(cpuMilli=16000, memoryMib=120000), maxRetryCount=0,
            maxRunDuration=f"{max_run_seconds}s"))],
        allocationPolicy=dict(serviceAccount=dict(email=service_account),
            instances=[dict(policy=dict(machineType="c4-highmem-16", provisioningModel="STANDARD",
                bootDisk=dict(sizeGb=boot_disk_gb, type="hyperdisk-balanced", image=VM_IMAGE)))]),
        logsPolicy=dict(destination="CLOUD_LOGGING"),
        labels=labels)
    return job
