"""Bounded Batch submission, pinned refit downloads and durable cloud execution."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import threading
from types import SimpleNamespace

import pytest

from hunl_escher import gtow_p32_cloud as cloud


def launcher():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "tools"))
    try:
        spec = importlib.util.spec_from_file_location("gtow_launcher", root / "tools/launch_gtow_p32.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def test_bundle_pins_batch_resources_and_source_without_credentials(tmp_path):
    plan = launcher().build_bundle(tmp_path / "launch", run_name="hunl-gtow-test")
    command = plan["commands"][-1]
    assert command[:4] == ["gcloud", "batch", "jobs", "submit"]
    assert command[command.index("--location") + 1] == "europe-west1"
    job = json.loads(Path(command[command.index("--config") + 1]).read_text())
    group, = job["taskGroups"]
    assert group["taskCount"] == group["parallelism"] == 1
    task = group["taskSpec"]
    assert task["maxRetryCount"] == 0 and task["maxRunDuration"] == "93600s"
    assert task["computeResource"] == {"cpuMilli": 2000, "memoryMib": 6144}
    policy = job["allocationPolicy"]["instances"][0]["policy"]
    assert policy["machineType"] == "c4-standard-2" and policy["provisioningModel"] == "STANDARD"
    assert policy["bootDisk"] == dict(sizeGb=30, type="hyperdisk-balanced",
        image="projects/batch-custom-image/global/images/family/batch-debian-12-official")
    assert job["logsPolicy"]["destination"] == "CLOUD_LOGGING"
    assert plan["backend"] == "google_cloud_batch"
    assert all("--if-generation-match=0" in c for c in plan["commands"][:-1])
    assert plan["pilot_hands"] == 1000 and plan["confirmation_hands"] == 20000
    script = tmp_path / "launch/run.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    text = script.read_text()
    assert "gtow_p32_cloud" in text and "shutdown" not in text
    assert "set -x" not in text and "source ./.env" not in text
    assert "sha256sum --check" in text
    assert "--kill-after=600s 25h" in text
    main, cleanup = task["runnables"]
    assert main["script"]["text"] == text and main["timeout"] == "91800s"
    assert cleanup["alwaysRun"] is True and cleanup["timeout"] == "900s"
    assert not main.get("ignoreExitStatus") and not cleanup.get("ignoreExitStatus")
    cleanup_script = tmp_path / "launch/publish.sh"
    subprocess.run(["bash", "-n", str(cleanup_script)], check=True)
    assert cleanup["script"]["text"] == cleanup_script.read_text()
    with tarfile.open(tmp_path / "launch/source.tar.gz") as archive:
        names = archive.getnames()
    assert "hunl_escher/gtow_p32_cloud.py" in names
    assert "hunl_escher/gtow_p32.py" in names
    assert not any("outputs/" in n or ".env" in n or ".git/" in n for n in names)
    with pytest.raises(FileExistsError):
        launcher().build_bundle(tmp_path / "launch", run_name="hunl-gtow-test")


def test_prepare_command_cannot_submit_and_remote_hash_checked(tmp_path, monkeypatch):
    module = launcher()
    def forbidden(*args, **kwargs):
        pytest.fail("A dry-run must not invoke gcloud")
    monkeypatch.setattr(module.subprocess, "run", forbidden)
    module.main(["--run-name", "hunl-gtow-test", "--output", str(tmp_path / "dry")])
    plan = json.loads((tmp_path / "dry/launch.json").read_text())
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=b"wrong"))
    with pytest.raises(ValueError, match="Remote refit manifest"):
        module.verify_remote_manifest(plan)


@pytest.mark.parametrize("script_name", ["run.sh", "publish.sh"])
@pytest.mark.parametrize("initial_path", [None, "", "/custom/cloud-sdk/bin"])
def test_batch_path_is_exported_to_child_processes(tmp_path, script_name, initial_path):
    launcher().build_bundle(tmp_path / "launch", run_name="hunl-gtow-test")
    script = (tmp_path / "launch" / script_name).read_text()
    setup = next(line for line in script.splitlines() if line.startswith("export PATH="))
    probe = shlex.join([sys.executable, "-I", "-c", "import os; print(os.environ['PATH'])"])
    env = {} if initial_path is None else {"PATH": initial_path}
    result = subprocess.run(["/bin/bash", "-eu", "-c", setup + "\n" + probe],
                            env=env, capture_output=True, text=True, check=True, timeout=15)
    entries = result.stdout.strip().split(":")
    assert entries[:6] == ["/usr/local/sbin", "/usr/local/bin", "/usr/sbin", "/usr/bin", "/sbin", "/bin"]
    assert "" not in entries
    if initial_path:
        assert initial_path in entries


@pytest.mark.parametrize("initial_path", [None, ""])
def test_real_venv_creation_with_minimal_batch_environment(tmp_path, initial_path):
    launcher().build_bundle(tmp_path / "launch", run_name="hunl-gtow-test")
    lines = (tmp_path / "launch/run.sh").read_text().splitlines()
    setup = next(line for line in lines if line.startswith("export PATH="))
    diagnostic = next(line for line in lines if line.startswith("/usr/bin/python3 -I -c "))
    creation = next(line for line in lines if line.startswith("/usr/bin/python3 -I -m venv "))
    venv = tmp_path / "bootstrap venv"
    # Execute the submitted commands, substituting only the local interpreter
    # and temporary output path. This is local regression, not GCP acceptance.
    script = "\n".join([setup, diagnostic, creation]).replace(
        "/usr/bin/python3", shlex.quote(sys.executable)).replace(
        "/opt/hunl/venv", shlex.quote(str(venv)))
    env = {"PYTHONHOME": "/nonexistent-test-python", "PYTHONPATH": "/nonexistent-test-python"}
    if initial_path is not None:
        env["PATH"] = initial_path
    result = subprocess.run(["/bin/bash", "-eu", "-c", script], env=env,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert Path(json.loads(result.stdout)["base_executable"]).is_absolute()
    check = subprocess.run([str(venv / "bin/python"), "-I", "-c",
                            "import sys; assert sys.prefix != sys.base_prefix"],
                           env=env, capture_output=True, text=True, timeout=15)
    assert check.returncode == 0, check.stderr


@pytest.mark.parametrize("failure_stage", ["", "system_packages", "python_venv"])
def test_bootstrap_reports_failed_stage_and_stops_before_play(tmp_path, failure_stage):
    launcher().build_bundle(tmp_path / "launch", run_name="hunl-gtow-test")
    script = (tmp_path / "launch/run.sh").read_text()
    body = "GTO_SETUP_STAGE=" + script.split("GTO_SETUP_STAGE=", 1)[1]
    mocks = 'mock_command() { if [[ "$GTO_SETUP_STAGE" == "$FAIL_STAGE" ]]; then return 37; fi; }\n'
    for command in ("apt-get", "mkdir", "cd", "gcloud", "sha256sum", "tar", "/usr/bin/python3",
                    "/opt/hunl/venv/bin/python", "timeout"):
        mocks += f'function {command}() {{ mock_command; }}\n'
    result = subprocess.run(["/bin/bash", "-Eeuo", "pipefail", "-c", mocks + body],
                            env={"FAIL_STAGE": failure_stage}, capture_output=True, text=True, timeout=15)
    if failure_stage:
        assert result.returncode == 37
        assert f"ERROR: stage={failure_stage} line=" in result.stderr and "exit_code=37" in result.stderr
        assert "GTO stage=benchmark_worker" not in result.stdout
    else:
        assert result.returncode == 0, result.stderr
        assert "GTO stage=benchmark_worker" in result.stdout


def test_package_install_waits_for_updater_without_upgrading_cloud_cli(tmp_path):
    launcher().build_bundle(tmp_path / "launch", run_name="hunl-gtow-test")
    script = (tmp_path / "launch/run.sh").read_text()
    install = next(line for line in script.splitlines() if line.startswith("apt-get ") and " install " in line)
    # Exercise the actual generated shell command against a package-manager stub.
    # A real dpkg lock-contention check is also run in the Batch diagnostic.
    mock = 'apt-get() { printf "%s\\n" "$@"; }\n'
    result = subprocess.run(["/bin/bash", "-eu", "-c", mock + install],
                            capture_output=True, text=True, check=True, timeout=15)
    args = result.stdout.splitlines()
    assert "DPkg::Lock::Timeout=600" in args
    assert "Acquire::Retries=3" in args and "--no-upgrade" in args
    assert "python3-venv" in args and "google-cloud-cli" in args


def test_refit_download_is_small_and_verified_before_loading(tmp_path, monkeypatch):
    data = b"policy bytes"
    digest = hashlib.sha256(data).hexdigest()
    manifest = json.dumps(dict(status="COMPLETE", smoke=False,
        results=[dict(checkpoint=f"{arm}_seed{seed}.pkl", sha256=digest)
                 for arm in ("uniform", "objective_mass") for seed in (1701, 1702)])).encode()
    calls = []
    def storage(*args):
        calls.append(args)
        Path(args[-1]).write_bytes(manifest if str(args[-1]).endswith("experiment.json") else data)
    monkeypatch.setattr(cloud, "storage", storage)
    monkeypatch.setattr(cloud, "verified_candidate", lambda root: ({}, {"policy_sha256": digest}))
    result = cloud.download_refit("gs://bucket/source", tmp_path / "fit",
                                 hashlib.sha256(manifest).hexdigest(), digest)
    assert len(calls) == 5 and len(list(result.glob("*.pkl"))) == 4
    assert all("/fit/" in args[1] for args in calls)
    with pytest.raises(ValueError, match="manifest checksum"):
        cloud.download_refit("gs://bucket/source", tmp_path / "bad", "0" * 64, digest)
    assert len(calls) == 6


@pytest.mark.parametrize("benchmark_status", ["COMPLETE", "INCOMPLETE_PROVIDER_ERROR", "EVALUATION_ERROR"])
def test_worker_publishes_outcomes_and_cleans_secret(tmp_path, monkeypatch, capsys, benchmark_status):
    transfers, runs = [], []
    monkeypatch.delenv("GTO_WIZARD_API_KEY", raising=False)
    monkeypatch.setattr(cloud, "storage", lambda *args: transfers.append(args))
    monkeypatch.setattr(cloud, "download_refit", lambda *args: tmp_path / "refit")
    monkeypatch.setattr(cloud, "read_secret", lambda *args: "SENSITIVE_TEST_KEY")
    def benchmark(root, output, **kwargs):
        assert os.environ["GTO_WIZARD_API_KEY"] == "SENSITIVE_TEST_KEY"
        runs.append(kwargs)
        result = {"status": benchmark_status}
        if result["status"] == "EVALUATION_ERROR":
            result.update(error_type="ProtocolError", reason="Cannot identify our seat without prior hand context",
                          completed_pilot_hands=1, completed_confirmation_hands=0)
        return result
    monkeypatch.setattr(cloud, "run_benchmark", benchmark)
    result = cloud.run_worker(tmp_path / "work", "gs://bucket/source", "gs://bucket/result",
                             "a" * 64, "b" * 64, "test-project", "test-secret")
    assert runs == [{"max_seconds": 86400}]
    assert result["status"] == benchmark_status
    assert "GTO_WIZARD_API_KEY" not in os.environ
    assert transfers[-1][-1] == "gs://bucket/result/cloud_status.json"
    assert "SENSITIVE_TEST_KEY" not in (tmp_path / "work/results/cloud_status.json").read_text()
    logs = capsys.readouterr().out
    assert "SENSITIVE_TEST_KEY" not in logs
    if benchmark_status == "EVALUATION_ERROR":
        assert "Cannot identify our seat" in logs
        assert result["benchmark"]["completed_pilot_hands"] == 1
        assert result["benchmark"]["reason"] == "Cannot identify our seat without prior hand context"


def test_secret_failure_is_sanitized_and_published_without_play(tmp_path, monkeypatch):
    transfers = []
    monkeypatch.setattr(cloud, "storage", lambda *args: transfers.append(args))
    monkeypatch.setattr(cloud, "download_refit", lambda *args: tmp_path / "refit")
    monkeypatch.setattr(cloud.subprocess, "run", lambda *a, **k:
                        SimpleNamespace(returncode=1, stdout="SENSITIVE_TEST_KEY", stderr="SENSITIVE_TEST_KEY"))
    monkeypatch.setattr(cloud, "run_benchmark", lambda *a, **k: pytest.fail("Must not play without secret"))
    with pytest.raises(RuntimeError, match="Secret Manager access") as error:
        cloud.run_worker(tmp_path / "work", "gs://bucket/source", "gs://bucket/result",
                         "a" * 64, "b" * 64, "test-project", "test-secret")
    assert "SENSITIVE_TEST_KEY" not in str(error.value)
    status = (tmp_path / "work/results/cloud_status.json").read_text()
    assert json.loads(status)["status"] == "FAILED" and "SENSITIVE_TEST_KEY" not in status
    assert transfers[-1][-1] == "gs://bucket/result/cloud_status.json"


def test_worker_rejects_source_overwrite_before_download(tmp_path):
    with pytest.raises(ValueError, match="separate"):
        cloud.run_worker(tmp_path / "work", "gs://bucket/source", "gs://bucket/source",
                         "a" * 64, "b" * 64, "test-project", "test-secret")
    assert not (tmp_path / "work").exists()


def test_periodic_publication_snapshots_complete_journal_lines(tmp_path, monkeypatch):
    published = threading.Event()
    snapshots = []
    def storage(*args):
        if args[0] == "rsync":
            journal = Path(args[-2]) / "benchmark/hands.jsonl"
            if journal.exists():
                snapshots.append(journal.read_bytes())
                published.set()
    monkeypatch.setattr(cloud, "storage", storage)
    with cloud.publication(tmp_path / "work", "gs://bucket/result", interval=0.02) as (output, status, stage):
        directory = output / "benchmark"
        directory.mkdir()
        journal = directory / "hands.jsonl"
        journal.write_bytes(b'{"hand": 1}\n{"hand":')
        (directory / "report.json.tmp").write_text("incomplete")
        stage("BENCHMARK")
        assert published.wait(5), "Expected a periodic upload while the worker was still running"
        assert snapshots[-1] == b'{"hand": 1}\n'
        with journal.open("ab") as stream:
            stream.write(b' 2}\n')
        stage("COMPLETE")
    assert snapshots[-1] == b'{"hand": 1}\n{"hand": 2}\n'
    assert not list((tmp_path / "work").glob("publication-*"))


def test_final_upload_failure_prevents_success_and_status_publication(tmp_path, monkeypatch):
    transfers = []
    def storage(*args):
        transfers.append(args)
        raise RuntimeError("Storage unavailable")
    monkeypatch.setattr(cloud, "storage", storage)
    with pytest.raises(RuntimeError, match="Storage unavailable"):
        with cloud.publication(tmp_path / "work", "gs://bucket/result") as (_, _, stage):
            stage("COMPLETE")
    assert len(transfers) == 1 and transfers[0][0] == "rsync"


def test_interruption_publishes_journal_and_failure_status(tmp_path, monkeypatch):
    transfers = []
    monkeypatch.setattr(cloud, "storage", lambda *args: transfers.append(args))
    with pytest.raises(KeyboardInterrupt):
        with cloud.publication(tmp_path / "work", "gs://bucket/result"):
            raise KeyboardInterrupt()
    status = json.loads((tmp_path / "work/results/cloud_status.json").read_text())
    assert status["status"] == "INTERRUPTED"
    assert transfers[-1][-1] == "gs://bucket/result/cloud_status.json"
