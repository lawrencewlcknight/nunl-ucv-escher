"""Separate bounded audit jobs, immutable positions and fail-closed policy inputs."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import pytest

from hunl_escher import position_audit_batch as worker
from hunl_escher.position_audit_config import (
    LIBRARY_PATH, LIBRARY_SHA256, frozen_library, policies)
from hunl_escher.strategic_audit.library import digest, validate_library

REPO = Path(__file__).resolve().parents[1]


def launcher():
    sys.path.insert(0, str(REPO / 'tools'))
    try:
        spec = importlib.util.spec_from_file_location('audit_launcher', REPO / 'tools/launch_position_audit.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(REPO / 'tools'))


def test_frozen_positions_match_original_p32_contract(tmp_path):
    library = validate_library(frozen_library())
    assert digest(library) == LIBRARY_SHA256
    assert len(library['positions']) == 526
    assert sum(p['partition'] == 'development' for p in library['positions']) == 374
    library['positions'][0]['hole'] = 'AsAh'
    path = tmp_path / 'changed.json'
    path.write_text(json.dumps(library))
    with pytest.raises(ValueError, match='unchanged'):
        frozen_library(path)


@pytest.mark.parametrize('hours', [56, 80])
def test_small_independent_batch_bundle(tmp_path, hours):
    module = launcher()
    root = tmp_path / 'bundle'
    plan = module.build_bundle(root, checkpoint_hours=hours, run_name=f'hunl-positions-p{hours}-test')
    assert [p['label'] for p in plan['policies']] == ['P32', f'P{hours}']
    assert not plan['training'] and not plan['external_benchmarks'] and not plan['assessment_unlocked']
    job = json.loads((root / 'job.json').read_text())
    task = job['taskGroups'][0]['taskSpec']
    assert task['maxRetryCount'] == 0 and task['maxRunDuration'] == '3600s'
    assert task['computeResource'] == dict(cpuMilli=2000, memoryMib=6144)
    machine = job['allocationPolicy']['instances'][0]['policy']
    assert machine['machineType'] == 'c4-standard-2' and machine['bootDisk']['sizeGb'] == 30
    script = (root / 'batch.sh').read_text()
    assert f'--checkpoint-hours {hours}' in script and '1200s' in script
    assert '.[parallel]' not in script and 'hunl_escher.position_audit_batch' in script
    subprocess.run(['bash', '-n', str(root / 'batch.sh')], check=True)
    with tarfile.open(root / 'source.tar.gz') as archive:
        name = 'hunl_escher/data/strategic_positions_v1.json'
        assert archive.extractfile(name).read() == LIBRARY_PATH.read_bytes()
        assert not any(n.endswith(('.pkl', '.pt', '.env')) or n.startswith('outputs/') for n in archive.getnames())
    assert plan['source_sha256'] == module.sha256(root / 'source.tar.gz')
    with pytest.raises(ValueError, match='independent'):
        module.build_bundle(tmp_path / 'bad', checkpoint_hours=hours,
                            run_name=policies(hours)[1]['uri'].rsplit('/', 1)[1])


@pytest.mark.parametrize('damage', [None, 'marker', 'checksum', 'payload'])
def test_policy_identity_checked_before_audit(tmp_path, monkeypatch, damage):
    spec = policies(80)[1]
    marker = dict(sha256=spec['policy_sha256'], outer_iteration=4160, active_seconds=288148.)
    if damage == 'marker':
        marker['outer_iteration'] = 4161
    calls, loads = [], []
    def download(*args):
        calls.append(args)
        destination = Path(args[-1])
        destination.write_bytes(json.dumps(marker).encode() if destination.suffix == '.json' else b'policy')
    monkeypatch.setattr(worker, 'storage', download)
    monkeypatch.setattr(worker, 'sha256_file', lambda path: 'wrong' if damage == 'checksum' else spec['policy_sha256'])
    def load(path):
        loads.append(path)
        return dict(outer_iteration=4161 if damage == 'payload' else 4160, training_elapsed_seconds=288148.)
    monkeypatch.setattr(worker, 'load_checkpoint_payload', load)
    if damage:
        with pytest.raises(ValueError):
            worker.fetch_policy(spec, tmp_path)
        if damage in ('marker', 'checksum'):
            assert not loads
    else:
        assert worker.fetch_policy(spec, tmp_path)[1] == marker
    assert all('/training/' not in str(c) for c in calls)


@pytest.mark.parametrize('hours', [56, 80])
@pytest.mark.parametrize('outcome', ['COMPLETE', 'INCOMPLETE_TIME_LIMIT', 'FAILED', 'bad_coverage'])
def test_worker_publishes_success_partial_and_failure(tmp_path, monkeypatch, hours, outcome):
    uploads = []
    monkeypatch.setattr(worker, 'storage', lambda *a, **kw: uploads.append(a))
    monkeypatch.setattr(worker, 'fetch_policy', lambda spec, dest: (dest / (spec['label'] + '.pkl'), {}))
    def audit(library, selected, output, **kwargs):
        assert digest(json.loads(library.read_text())) == LIBRARY_SHA256
        assert [label for label, path in selected] == ['P32', f'P{hours}']
        assert kwargs == dict(partition='development', unlock_assessment=False, max_seconds=600, threads=1)
        output.mkdir()
        (output / 'report.html').write_text('human-readable report')
        if outcome == 'FAILED':
            raise ValueError('audit failure')
        count = 1 if outcome in ('bad_coverage', 'INCOMPLETE_TIME_LIMIT') else 374
        return dict(status='COMPLETE' if outcome == 'bad_coverage' else outcome,
                    expected_positions_per_policy=374, elapsed_seconds=1,
                    summary=dict(policies={p[0]: dict(completed_positions=count) for p in selected}))
    monkeypatch.setattr(worker, 'run', audit)
    root = tmp_path / 'work'
    if outcome in ('FAILED', 'bad_coverage'):
        with pytest.raises(ValueError):
            worker.run_worker(root, 'gs://test/audit', hours)
        expected = 'FAILED'
    else:
        expected = outcome
        assert worker.run_worker(root, 'gs://test/audit', hours)['status'] == expected
    assert json.loads((root / 'results/batch_status.json').read_text())['status'] == expected
    assert uploads[-2][0] == 'rsync' and uploads[-1][-1].endswith('/batch_status.json')
    assert (root / 'results/audit/report.html').exists()


@pytest.mark.parametrize('failure', ['duplicate', 'tampered', 'network', 'source'])
def test_submission_rejects_before_upload(tmp_path, monkeypatch, failure):
    module = launcher()
    plan = module.build_bundle(tmp_path / 'bundle', checkpoint_hours=80, run_name='hunl-positions-test')
    if failure == 'tampered':
        (tmp_path / 'bundle/job.json').write_text('{}')
    calls = []
    def command(args, **kwargs):
        calls.append(args)
        if failure == 'network':
            raise subprocess.CalledProcessError(1, args)
        if args[1:4] == ['batch', 'jobs', 'list']:
            return SimpleNamespace(stdout=json.dumps([dict(name='jobs/' + plan['job_name'])] if failure == 'duplicate' else []))
        if args[1:3] == ['storage', 'cat']:
            return SimpleNamespace(stdout=json.dumps(dict(sha256='wrong', outer_iteration=0)))
        pytest.fail('Must reject before any upload or paid submission')
    monkeypatch.setattr(module.subprocess, 'run', command)
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        module.submit(tmp_path / 'bundle')
    assert not any(c[1:3] == ['storage', 'cp'] for c in calls)
