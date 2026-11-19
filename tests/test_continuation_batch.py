"""Real bounded migration/continuation plus Batch deployment contract checks."""
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch

from hunl_escher.corrected_continuation import fingerprint, patch_trainer, rebind_metadata
from hunl_escher.continuation import run_continuation
from hunl_escher.continuation_cloud import extract_source
from hunl_escher.experiment import publish_policy
from hunl_escher.training_state import read_training_state

REPO = Path(__file__).resolve().parents[1]


def launcher():
    sys.path.insert(0, str(REPO / "tools"))
    try:
        spec = importlib.util.spec_from_file_location("batch_launcher", REPO / "tools/launch_continuation_batch.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def equal(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, float) and math.isnan(a):
        assert math.isnan(b)
    else:
        assert a == b


def test_migration_preserves_all_learning_state_and_continues(tmp_path):
    module = launcher()
    module.source_archive(REPO, tmp_path / "source.tar.gz")
    archived = tmp_path / "legacy"
    extract_source(tmp_path / "source.tar.gz", archived)
    shutil.copyfile(REPO / "tests/fixtures/p56_legacy_holdem_fitting.py.txt",
                    archived / "unbiased_escher/holdem_fitting.py")
    (archived / "unbiased_escher/policy_sampling.py").unlink()
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    env = {**os.environ, "PYTHONPATH": str(archived), "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    result = subprocess.run([sys.executable, "-m", "hunl_escher.train", "--smoke", "--iterations", "1",
        "--workers", "1", "--threads", "1", "--seed", "123", "--policy-every", "0",
        "--policy-seconds", "1000000000", "--output", str(baseline / "training")],
        cwd=archived, env=env, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr[-4000:]
    publish_policy(baseline / "training", baseline, "P56")
    original = read_training_state(baseline / "training/state_000001")
    original_metadata = (baseline / "training/state_000001/metadata.pt").read_bytes()
    patched = tmp_path / "patched"
    shutil.copytree(archived, patched)
    before = {str(p.relative_to(patched)): p.read_bytes() for p in patched.rglob("*.py")}
    changes = patch_trainer(patched, 123)
    assert set(changes) == {"unbiased_escher/holdem_fitting.py", "unbiased_escher/policy_sampling.py"}
    for name, value in before.items():
        if name not in changes:
            assert (patched / name).read_bytes() == value
    migrated = tmp_path / "migrated"
    rebind_metadata(baseline, migrated, fingerprint(patched))
    state = read_training_state(migrated / "training/state_000001")  # verifies every shard hash
    assert (baseline / "training/state_000001/metadata.pt").read_bytes() == original_metadata
    assert original["repository_commit"] != state["repository_commit"]
    for key in original:
        if key != "repository_commit":
            equal(original[key], state[key])
    assert (migrated / "P56.pkl").read_bytes() == (baseline / "P56.pkl").read_bytes()
    report = run_continuation(migrated, patched, tmp_path / "continued", baseline_hours=56,
        stage_seconds=.001, wall_seconds=240, allow_smoke=True, verify_each_stage=True)
    assert report["status"] == "TRAINING_COMPLETE" and report["final_recovery_verified"]
    assert list(report["checkpoints"]) == ["P56", "P62", "P68", "P74", "P80"]
    for label in ("P62", "P68", "P74", "P80"):
        diag = report["checkpoints"][label]["metrics"]["policy_validation"]
        assert diag["group_sampling"] == "objective_mass"
        assert diag["policy_sampling_seed"] == [123, report["checkpoints"][label]["outer_iteration"]]
    for label in ("P56", "P62", "P68", "P74", "P80"):
        assert json.loads((tmp_path / "continued" / f"reload-{label}.json").read_text())["status"] == "VERIFIED"
    from hunl_escher.continuation_evaluate import run_suite
    evaluation = run_suite(tmp_path / "continued", tmp_path / "evaluation", workers=1, max_seconds=180,
                           smoke=True, external_benchmarks=False)
    assert evaluation["status"] == "COMPLETE"
    assert evaluation["primary"] == "P80-vs-P56"
    assert len(evaluation["head_to_head"]) == 5
    assert "External benchmarks are disabled" in (tmp_path / "evaluation/continuation_report.md").read_text()
    with pytest.raises(ValueError, match="checksum"):
        patch_trainer(patched, 123)


def test_batch_contract_and_shell(tmp_path):
    module = launcher()
    bundle = tmp_path / "bundle"
    plan = module.build_bundle(bundle, project="test", region="europe-west1", bucket="gs://test/hunl",
        service_account="runner@test.iam.gserviceaccount.com", run_name="hunl-batch-test")
    job = json.loads((bundle / "job.json").read_text())
    spec = job["taskGroups"][0]["taskSpec"]
    assert spec["maxRetryCount"] == 0 and spec["maxRunDuration"] == "129600s"
    assert spec["computeResource"] == dict(cpuMilli=16000, memoryMib=120000)
    assert job["allocationPolicy"]["instances"][0]["policy"]["bootDisk"]["type"] == "hyperdisk-balanced"
    assert "batch-cos" in plan["vm_image"] and "@sha256:" in plan["python_container"]
    assert plan["policy_group_sampling"] == "objective_mass"
    assert plan["labels"] == ["P56", "P62", "P68", "P74", "P80"]
    script = (bundle / "batch.sh").read_text()
    assert "hunl_escher.continuation_batch" in script and "shutdown" not in script
    assert "pip==25.3" in script and "--no-deps --only-binary=:all: torch==2.7.0+cpu" in script
    assert "APT::Update::Error-Mode=any" in script
    assert "free -m || true" in script and "python3-venv procps" in script
    subprocess.run(["bash", "-n", str(bundle / "batch.sh")], check=True)
    assert plan['external_benchmarks'] is False
    assert job['allocationPolicy']['instances'][0]['policy']['machineType'] == 'c4-highmem-16'
    assert job['allocationPolicy']['location']['allowedLocations'] == ['regions/europe-west1']


@pytest.mark.parametrize('failure', [False, True])
def test_worker_is_self_contained_and_publishes_failure(tmp_path, monkeypatch, failure):
    from contextlib import contextmanager
    from hunl_escher import continuation_batch as worker
    uploads, stages, syncs = [], [], []
    @contextmanager
    def publisher(path, uri, **kwargs):
        yield lambda **kw: syncs.append((path, kw))
        syncs.append((path, 'final'))
    monkeypatch.setattr(worker, 'publisher', publisher)
    def download(uri, path, digest, hours):
        assert uri == worker.SOURCE_URI and digest == worker.P56_SHA256 and hours == 56
        path.mkdir()
        return path, path
    monkeypatch.setattr(worker, 'download_source', download)
    def migrate(source, trainer, root):
        if failure:
            raise ValueError('Rejected source')
        (source / 'training_source.tar.gz').write_bytes(b'archive')
        return source, trainer
    monkeypatch.setattr(worker, 'prepare_corrected_source', migrate)
    monkeypatch.setattr(worker, 'storage', lambda *args: uploads.append(args))
    def train(source, trainer, output, **kwargs):
        stages.append('train')
        assert kwargs == dict(baseline_hours=56, stage_seconds=21600,
            wall_seconds=108000, upload_uri='gs://test/new', verify_each_stage=True)
        output.mkdir()
        return dict(labels=['P56', 'P62', 'P68', 'P74', 'P80'], training_source_fingerprint='patched',
            checkpoints={label: dict(sha256='hash', outer_iteration=i, metrics=dict(policy_validation=dict(
                group_sampling='objective_mass', policy_sampling_seed=[20261115, i])))
                for i, label in enumerate(['P56', 'P62', 'P68', 'P74', 'P80'])})
    monkeypatch.setattr(worker, 'run_continuation', train)
    def evaluate(root, output, **kwargs):
        assert stages == ['train']
        assert kwargs == dict(workers=8, max_seconds=14400, external_benchmarks=False)
        stages.append('evaluate')
        return dict(status='COMPLETE')
    monkeypatch.setattr(worker, 'run_suite', evaluate)
    root = tmp_path / 'work'
    if failure:
        with pytest.raises(ValueError, match='Rejected source'):
            worker.run_worker(root, 'gs://test/new')
        assert stages == [] and uploads == []
        assert json.loads((root / 'status/batch_status.json').read_text())['status'] == 'FAILED'
    else:
        assert worker.run_worker(root, 'gs://test/new')['status'] == 'COMPLETE'
        assert stages == ['train', 'evaluate']
        assert (root / 'run/resume.json').exists()
        assert not (root / 'run/slumbot_ready.json').exists()
        assert len(syncs) >= 2


def test_duplicate_batch_submission_stops_before_upload(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import hashlib
    module = launcher()
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout=json.dumps([dict(name='projects/test/locations/europe-west1/jobs/same')]))
    monkeypatch.setattr(module.subprocess, 'run', run)
    (tmp_path / 'source.tar.gz').write_bytes(b'archive')
    with pytest.raises(ValueError, match='already exists'):
        module.submit(tmp_path, dict(project='test', region='europe-west1', job_name='same',
            source_sha256=hashlib.sha256(b'archive').hexdigest()))
    assert len(calls) == 1 and calls[0][1:4] == ['batch', 'jobs', 'list']
