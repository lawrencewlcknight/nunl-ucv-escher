"""Bounded separate Batch jobs and small, checksum-verified policy downloads."""
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import pytest

from hunl_escher import p128_benchmark as benchmark
from hunl_escher import p128_benchmark_cloud as cloud
from hunl_escher.checkpointing import sha256_file


def launcher():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / 'tools'))
    try:
        spec = importlib.util.spec_from_file_location('p128_launcher', root / 'tools/launch_p128_benchmark.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


@pytest.mark.parametrize('hours,provider,counts', [(128, 'gtow', (0, 8000)),
    (128, 'slumbot', (0, 20000)), (200, 'slumbot', (0, 8000))])
def test_separate_bounded_batch_bundle(tmp_path, hours, provider, counts):
    mod = launcher()
    output = tmp_path / provider
    plan = mod.build_bundle(output, provider=provider, run_name=f'hunl-p{hours}-' + provider,
                            policy_hours=hours)
    assert (plan['pilot_hands'], plan['confirmation_hands']) == counts
    assert plan['total_hand_cap'] == sum(counts)
    profile = benchmark.benchmark_profile(hours)
    assert plan['expected_policy_sha256'] == profile['policy_sha256']
    assert plan['expected_manifest_sha256'] == profile['manifest_sha256']
    assert plan['policy_hours'] == hours
    job = json.loads((output / 'job.json').read_text())
    task = job['taskGroups'][0]['taskSpec']
    assert task['maxRetryCount'] == 0 and task['maxRunDuration'] == '93600s'
    assert job['taskGroups'][0]['parallelism'] == 1
    assert job['allocationPolicy']['instances'][0]['policy']['machineType'] == 'c4-standard-2'
    assert task['runnables'][1]['alwaysRun']
    assert task['computeResource']['memoryMib'] == 6144
    for name in ('run.sh', 'publish.sh'):
        subprocess.run(['bash', '-n', str(output / name)], check=True)
        assert 'export PATH=' in (output / name).read_text()
    script = (output / 'run.sh').read_text()
    assert '--provider ' + provider in script and 'DPkg::Lock::Timeout=600' in script
    assert f'--policy-hours {hours}' in script
    assert '--kill-after=600s 25h' in script and 'hunl_escher.p128_benchmark_cloud' in script
    assert 'OMP_NUM_THREADS=1' in script
    assert all('--if-generation-match=0' in c for c in plan['commands'][:-1])
    with tarfile.open(output / 'source.tar.gz') as archive:
        names = archive.getnames()
    assert 'hunl_escher/p128_benchmark.py' in names
    assert not any(n.startswith('outputs/') or '.env' in n or n.endswith('.pkl') for n in names)


def test_preparation_is_offline_and_changed_bundle_cannot_submit(tmp_path, monkeypatch):
    mod = launcher()
    def forbidden(*a, **k):
        pytest.fail('Preparation or failed integrity check must not contact GCP')
    monkeypatch.setattr(mod.subprocess, 'run', forbidden)
    mod.main(['--provider', 'gtow', '--run-name', 'hunl-p128-test', '--output', str(tmp_path / 'bundle')])
    (tmp_path / 'bundle/job.json').write_text('{}')
    with pytest.raises(ValueError, match='bundle changed'):
        mod.submit_prepared(tmp_path / 'bundle')


@pytest.mark.parametrize('hours', [128, 200])
def test_download_only_three_required_artifacts(tmp_path, monkeypatch, hours):
    source = tmp_path / 'source'
    source.mkdir()
    label = f'P{hours}'
    for name in (label + '.pkl', label + '.json', 'continuation.json'):
        (source / name).write_text(name)
    profile = dict(cloud.benchmark_profile(hours), uri='gs://test/source',
                   policy_sha256=sha256_file(source / (label + '.pkl')),
                   manifest_sha256=sha256_file(source / 'continuation.json'))
    monkeypatch.setattr(cloud, 'benchmark_profile', lambda hours=128: profile)
    monkeypatch.setattr(cloud, 'verified_candidate', lambda root, **kwargs: None)
    seen = []
    def transfer(operation, uri, destination):
        assert operation == 'cp'
        name = uri.rsplit('/', 1)[1]
        seen.append(name)
        shutil.copyfile(source / name, destination)
    monkeypatch.setattr(cloud, 'storage', transfer)
    cloud.download_policy('gs://test/source', tmp_path / 'download', hours)
    assert seen == ['continuation.json', label + '.json', label + '.pkl']
    seen.clear()
    (source / 'continuation.json').write_text('changed')
    with pytest.raises(ValueError, match='checksum mismatch'):
        cloud.download_policy('gs://test/source', tmp_path / 'rejected', hours)
    assert seen == ['continuation.json']


def test_p200_rejects_changed_hand_budget_before_submission(tmp_path, monkeypatch):
    mod = launcher()
    plan = mod.build_bundle(tmp_path / 'bundle', provider='slumbot', policy_hours=200,
                            run_name='hunl-p200-test')
    plan['confirmation_hands'] = 20000
    (tmp_path / 'bundle/launch.json').write_text(json.dumps(plan))
    monkeypatch.setattr(mod.subprocess, 'run', lambda *a, **k: pytest.fail('Must reject before network'))
    with pytest.raises(ValueError, match='pinned benchmark'):
        mod.submit_prepared(tmp_path / 'bundle')


def test_p200_duplicate_job_cannot_consume_more_hands(tmp_path, monkeypatch):
    from types import SimpleNamespace
    mod = launcher()
    mod.build_bundle(tmp_path / 'bundle', provider='slumbot', policy_hours=200,
                     run_name='hunl-p200-test')
    monkeypatch.setattr(mod, 'verify_remote_manifest', lambda plan: None)
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        assert command[1:4] == ['batch', 'jobs', 'list']
        return SimpleNamespace(stdout=json.dumps([{'name': 'projects/test/locations/europe-west1/jobs/hunl-p200-test'}]))
    monkeypatch.setattr(mod.subprocess, 'run', run)
    with pytest.raises(ValueError, match='already exists'):
        mod.submit_prepared(tmp_path / 'bundle')
    assert len(calls) == 1


@pytest.mark.parametrize('provider', ['gtow', 'slumbot'])
def test_worker_secret_is_gto_only_and_failures_publish(tmp_path, monkeypatch, provider):
    published = []
    @contextmanager
    def publish(workspace, uri):
        output = Path(workspace) / 'results'
        output.mkdir(parents=True)
        status = {}
        try:
            yield output, status, lambda value: status.update(status=value)
        finally:
            published.append(status.copy())
    monkeypatch.setattr(cloud, 'publication', publish)
    monkeypatch.setattr(cloud, 'download_policy', lambda *args: tmp_path / 'policy')
    calls = []
    def secret(*args):
        calls.append(args)
        return 'test-only-key'
    monkeypatch.setattr(cloud, 'read_secret', secret)
    monkeypatch.setenv('GTO_WIZARD_API_KEY', 'original')
    def run(*args, **kwargs):
        import os
        assert kwargs['provider'] == provider
        assert os.environ['GTO_WIZARD_API_KEY'] == ('test-only-key' if provider == 'gtow' else 'original')
        return dict(status='INCOMPLETE_PROVIDER_ERROR', completed_pilot_hands=0, completed_confirmation_hands=3)
    monkeypatch.setattr(cloud, 'run_benchmark', run)
    result = cloud.run_worker(tmp_path / provider, 'gs://test/source', 'gs://test/' + provider,
                             provider, 'project', 'secret-name')
    assert result['status'] == 'INCOMPLETE_PROVIDER_ERROR' and len(published) == 1
    assert len(calls) == (1 if provider == 'gtow' else 0)
    import os
    assert os.environ['GTO_WIZARD_API_KEY'] == 'original'
    assert 'test-only-key' not in json.dumps(published)
