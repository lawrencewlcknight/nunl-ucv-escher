import json
from pathlib import Path
import subprocess

import pytest

from hunl_escher.continuation_cloud import stage_reports
from hunl_escher.experiment import prune_cloud_recoveries
from hunl_escher.recover_continuation import validate_completed_training


def test_report_staging_cannot_publish_checkpoint_or_advance_pointer(tmp_path):
    source, target = tmp_path / 'run', tmp_path / 'reports'
    names = ['training/state_001658/manifest.json', 'training/state_001658/metadata.pt',
             'training/latest.json', 'training/manifest.json', 'P32.pkl', 'P32.json',
             'cloud_checkpoint.json', 'training/policy_001658.pkl',
             'continuation.json', 'training/metrics_001658.json', 'evaluation/report.json',
             'evaluation/hands.jsonl', 'train-P32.log']
    for name in names:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('data')
    stage_reports(source, target, training=True, final=False)
    assert {str(p.relative_to(target)) for p in target.rglob('*') if p.is_file()} == {
        'continuation.json', 'training/metrics_001658.json', 'evaluation/report.json', 'train-P32.log'}
    stage_reports(source, target, training=True, final=True)
    assert (target / 'evaluation/hands.jsonl').exists()
    assert not (target / 'training/state_001658').exists()


def test_prune_all_older_run_states_preserves_newer_and_other_prefixes(monkeypatch):
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout='\n'.join([
            'gs://bucket/run/training/state_000100/', 'gs://bucket/run/training/state_000200/',
            'gs://bucket/run/training/state_000300/', 'gs://bucket/run/training/state_000400/',
            'gs://bucket/original/training/state_000001/', 'gs://bucket/run/training/latest.json',
            'gs://bucket/run/training/state_000200/arrays/']), stderr='')
    monkeypatch.setattr(subprocess, 'run', run)
    assert prune_cloud_recoveries('gs://bucket/run', 'state_000300') == ['state_000100', 'state_000200']
    assert len(calls) == 3
    assert all(c[2] == 'rm' for c in calls[1:])


def test_recovery_rejects_incomplete_training_before_remote_actions(tmp_path):
    (tmp_path / 'training').mkdir()
    (tmp_path / 'training/latest.json').write_text(json.dumps({'state': 'state_001658', 'iteration': 1658}))
    (tmp_path / 'continuation.json').write_text(json.dumps({'smoke': False, 'checkpoints': {'P8': {}}}))
    with pytest.raises(ValueError, match='five'):
        validate_completed_training(tmp_path)


@pytest.mark.parametrize('cloud_fails', [False, True])
def test_recovery_never_publishes_or_prunes_before_cloud_verification(tmp_path, monkeypatch, cloud_fails):
    from contextlib import contextmanager
    from hunl_escher import recover_continuation as recovery
    root = tmp_path / 'run'
    (root / 'training').mkdir(parents=True)
    report = {'status': 'FAILED', 'checkpoints': {'P32': {'sha256': 'a'*64}}}
    for name, data in [('continuation.json', report), ('launch.json', {'source_uri': 'gs://b/original'}),
                       ('training/manifest.json', {}), ('training/latest.json', {})]:
        (root / name).write_text(json.dumps(data))
    calls = []
    monkeypatch.setattr(recovery, 'validate_completed_training', lambda _: (report, {'state': 'state_001658'}))
    @contextmanager
    def publisher(*a, **k):
        yield lambda **kw: None
    monkeypatch.setattr(recovery, 'publisher', publisher)
    monkeypatch.setattr(recovery, 'run_transfer', lambda *a, **k: None)
    def verify(source, training, policy, output):
        calls.append('cloud' if training.name == 'cloud-verification' else 'disk')
        if cloud_fails and calls[-1] == 'cloud':
            raise ValueError('Cloud checksum mismatch')
    monkeypatch.setattr(recovery, 'verify', verify)
    monkeypatch.setattr(recovery, 'upload_checkpoint', lambda *a: calls.append('publish_and_prune'))
    monkeypatch.setattr(recovery, 'run_suite', lambda *a, **k: {'status': 'COMPLETE'})
    if cloud_fails:
        with pytest.raises(ValueError, match='checksum'):
            recovery.recover(tmp_path, 'gs://b/run')
        assert calls == ['disk', 'cloud']
        assert json.loads((root / 'continuation.json').read_text())['status'] == 'FAILED'
        assert not (root / 'slumbot_ready.json').exists()
    else:
        recovery.recover(tmp_path, 'gs://b/run')
        assert calls == ['disk', 'cloud', 'publish_and_prune']
        assert json.loads((root / 'continuation.json').read_text())['final_recovery_verified']
        assert (root / 'slumbot_ready.json').exists()
