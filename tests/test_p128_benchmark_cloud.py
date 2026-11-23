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


@pytest.mark.parametrize('provider,counts', [('gtow', (0, 8000)), ('slumbot', (0, 20000))])
def test_separate_bounded_batch_bundle(tmp_path, provider, counts):
    mod = launcher()
    output = tmp_path / provider
    plan = mod.build_bundle(output, provider=provider, run_name='hunl-p128-' + provider)
    assert (plan['pilot_hands'], plan['confirmation_hands']) == counts
    assert plan['total_hand_cap'] == sum(counts)
    assert plan['expected_policy_sha256'] == benchmark.POLICY_SHA256
    assert plan['expected_manifest_sha256'] == benchmark.MANIFEST_SHA256
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


def test_download_only_three_required_artifacts(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    source.mkdir()
    for name in ('P128.pkl', 'P128.json', 'continuation.json'):
        (source / name).write_text(name)
    monkeypatch.setattr(cloud, 'POLICY_SHA256', sha256_file(source / 'P128.pkl'))
    monkeypatch.setattr(cloud, 'MANIFEST_SHA256', sha256_file(source / 'continuation.json'))
    monkeypatch.setattr(cloud, 'verified_candidate', lambda root: None)
    seen = []
    def transfer(operation, uri, destination):
        assert operation == 'cp'
        name = uri.rsplit('/', 1)[1]
        seen.append(name)
        shutil.copyfile(source / name, destination)
    monkeypatch.setattr(cloud, 'storage', transfer)
    cloud.download_policy('gs://test/source', tmp_path / 'download')
    assert seen == ['continuation.json', 'P128.json', 'P128.pkl']
    seen.clear()
    (source / 'continuation.json').write_text('changed')
    with pytest.raises(ValueError, match='checksum mismatch'):
        cloud.download_policy('gs://test/source', tmp_path / 'rejected')
    assert seen == ['continuation.json']


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
