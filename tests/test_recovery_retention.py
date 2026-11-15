"""Latest-only recovery retention and publication failure boundaries."""
import json
import numpy as np
import pytest
import torch

from hunl_escher import train, state_storage
from hunl_escher.recovery_retention import prune_recovery_checkpoints
from hunl_escher.training_state import read_training_state


@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)


def test_prune_all_older_recoveries_only_after_publication(tmp_path):
    for iteration in (1, 2, 3, 4):
        state_storage.save_sharded_training_state(tmp_path / f"state_{iteration:06d}",
            {"type": state_storage.STATE_TYPE, "array": np.arange(8)})
    policy = tmp_path / "policy_000001.pkl"
    policy.write_bytes(b"keep this policy")
    (tmp_path / "metrics_000001.json").write_text('{}')
    unrelated = tmp_path / "state_000000"
    unrelated.mkdir()
    (unrelated / "notes.txt").write_text('not a recovery')
    # Never follow or remove a directory symlink while pruning.
    linked = tmp_path / "state_0000000"
    linked.symlink_to(unrelated, target_is_directory=True)
    train.write_json(tmp_path / "latest.json", {"state": "state_000002"})
    with pytest.raises(ValueError, match="Publish"):
        prune_recovery_checkpoints(tmp_path, "state_000003")
    assert (tmp_path / "state_000001").is_dir()
    train.write_json(tmp_path / "latest.json", {"state": "state_000003"})
    assert prune_recovery_checkpoints(tmp_path, "state_000003") == ['state_000001', 'state_000002']
    assert (tmp_path / "state_000003").is_dir()
    assert (tmp_path / "state_000004").is_dir()  # Not older than the published save.
    assert policy.read_bytes() == b"keep this policy"
    assert (tmp_path / "metrics_000001.json").exists()
    assert (unrelated / "notes.txt").read_text() == 'not a recovery'
    assert linked.is_symlink()


@pytest.mark.parametrize("failure", ["write", "verification", "publication"])
def test_failed_new_checkpoint_preserves_previous_recovery(tmp_path, monkeypatch, failure):
    if failure == "write":
        original = train.save_training_state
        def save(path, payload):
            if path.name == 'state_000002':
                raise OSError('injected failure')
            return original(path, payload)
        monkeypatch.setattr(train, 'save_training_state', save)
    elif failure == "verification":
        original = state_storage.training_state_storage_info
        def verify(path, **kwargs):
            if path.name == 'state_000002':
                raise OSError('injected failure')
            return original(path, **kwargs)
        monkeypatch.setattr(state_storage, 'training_state_storage_info', verify)
    else:
        original = train.write_json
        def publish(path, value):
            if path.name == 'latest.json' and value['iteration'] == 2:
                raise OSError('injected failure')
            return original(path, value)
        monkeypatch.setattr(train, 'write_json', publish)
    with pytest.raises(OSError, match='injected failure'):
        train.main(['--smoke', '--iterations', '2', '--output', str(tmp_path)])
    assert json.loads((tmp_path / 'latest.json').read_text())['state'] == 'state_000001'
    payload = read_training_state(tmp_path / 'state_000001')
    assert payload['solver']['num_iteration'] == 1
    assert (tmp_path / 'policy_000001.pkl').is_file()


def test_runner_keeps_latest_recovery_and_all_policies_and_resumes(tmp_path):
    args = ['--smoke', '--output', str(tmp_path)]
    train.main([*args, '--iterations', '3'])
    assert sorted(p.name for p in tmp_path.glob('state_*')) == ['state_000003']
    assert len(list(tmp_path.glob('policy_*.pkl'))) == 3
    metrics = json.loads((tmp_path / 'metrics_000003.json').read_text())
    assert metrics['recovery_checkpoints_deleted'] == ['state_000002']
    train.main([*args, '--iterations', '4', '--resume'])
    assert sorted(p.name for p in tmp_path.glob('state_*')) == ['state_000004']
    assert len(list(tmp_path.glob('policy_*.pkl'))) == 4
    assert read_training_state(tmp_path / 'state_000004')['solver']['num_iteration'] == 4
