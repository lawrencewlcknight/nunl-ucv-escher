import json
import subprocess

import pytest

from hunl_escher import storage_transfer
from hunl_escher.experiment import upload_checkpoint


def setup_run(root):
    (root/'training').mkdir()
    (root/'training/latest.json').write_text(json.dumps({'state': 'state_000507'}))
    (root/'cloud_checkpoint.json').write_text(json.dumps({'state': 'state_000301'}))


def test_503_retries_complete_before_pointer_and_policy_publication(tmp_path, monkeypatch):
    setup_run(tmp_path)
    calls, sleeps = [], []
    def run(command, **kwargs):
        calls.append(command)
        if len(calls) == 1:
            raise subprocess.CalledProcessError(1, command, stderr='ERROR: HTTPError 503\n')
        return subprocess.CompletedProcess(command, 0, stderr='')
    monkeypatch.setattr(subprocess, 'run', run)
    monkeypatch.setattr(storage_transfer.time, 'sleep', sleeps.append)
    upload_checkpoint(tmp_path, 'gs://bucket/run', 'P14')
    assert calls[0] == calls[1] and calls[0][2] == 'rsync'
    assert sleeps == [5]
    assert [c[-1] for c in calls[2:]] == [
        'gs://bucket/run/training/manifest.json', 'gs://bucket/run/training/latest.json',
        'gs://bucket/run/training/state_000301', 'gs://bucket/run/P14.pkl', 'gs://bucket/run/P14.json']
    assert json.loads((tmp_path/'cloud_checkpoint.json').read_text())['state'] == 'state_000507'


@pytest.mark.parametrize('message,count', [('ERROR: HTTPError 503', 4), ('ERROR: HTTPError 403', 1)])
def test_failed_upload_never_prunes_or_publishes(tmp_path, monkeypatch, message, count):
    setup_run(tmp_path)
    calls = []
    def fail(command, **kwargs):
        calls.append(command)
        raise subprocess.CalledProcessError(1, command, stderr=message)
    monkeypatch.setattr(subprocess, 'run', fail)
    monkeypatch.setattr(storage_transfer.time, 'sleep', lambda _: None)
    with pytest.raises(subprocess.CalledProcessError):
        upload_checkpoint(tmp_path, 'gs://bucket/run', 'P14')
    assert len(calls) == count and all(c[2] == 'rsync' for c in calls)
    assert json.loads((tmp_path/'cloud_checkpoint.json').read_text())['state'] == 'state_000301'


def test_total_deadline_bounds_retries(monkeypatch):
    clock, calls = [0.], []
    monkeypatch.setattr(storage_transfer.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(storage_transfer.time, 'sleep', lambda delay: clock.__setitem__(0, clock[0]+delay))
    def fail(command, **kwargs):
        calls.append(kwargs['timeout'])
        clock[0] += 2
        raise subprocess.CalledProcessError(1, command, stderr='HTTP 503')
    monkeypatch.setattr(subprocess, 'run', fail)
    with pytest.raises(subprocess.CalledProcessError):
        storage_transfer.run_transfer(['gcloud','storage','rsync','a','b'], timeout=10)
    assert calls == [10, 3]


def test_timeouts_and_transient_statuses_only():
    assert storage_transfer.transient_error(subprocess.TimeoutExpired('gcloud', 1))
    for code in (408, 429, 500, 502, 503, 504):
        assert storage_transfer.transient_error(subprocess.CalledProcessError(1, 'gcloud', stderr=f'HTTPError {code}'))
    assert not storage_transfer.transient_error(subprocess.CalledProcessError(1, 'gcloud', stderr='Missing state_000503'))
    with pytest.raises(ValueError, match='restricted'):
        storage_transfer.run_transfer(['gcloud','storage','rm','a'], timeout=10)
