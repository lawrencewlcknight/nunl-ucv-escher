"""Reject wrong corrected-baseline provenance before starting an expensive continuation."""
import json

import pytest

from hunl_escher.continuation_sources import source_profile, validate_source_metadata


def metadata(hours=80):
    p = source_profile(hours)
    iteration, label = p["iteration"], f"P{hours}"
    state = f"state_{iteration:06d}"
    marker = dict(sha256=p['policy_sha256'], outer_iteration=iteration, recovery=state,
        metrics=dict(policy_validation=dict(group_sampling='objective_mass',
                                           policy_sampling_seed=[20261115, iteration])))
    return {
        'continuation.json': dict(status='TRAINING_COMPLETE', final_recovery_verified=True,
                                 labels=[f'P{hours-24+6*i}' for i in range(5)], checkpoints={label: marker}),
        f'{label}.json': marker,
        'training/latest.json': dict(state=state, iteration=iteration),
        'training/manifest.json': dict(source_fingerprint=p['fingerprint'], workers=8, threads=8,
                                      seed=20261115, smoke=False),
        f'reload-{label}.json': dict(status='VERIFIED', iteration=iteration, source_fingerprint=p['fingerprint']),
        'resume.json': dict(source_fingerprint=p['fingerprint'], policy_group_sampling='objective_mass',
            training_source_archive_sha256=p['archive_sha256'],
            training_source_archive=p['uri'] + '/training_source.tar.gz'),
    }


@pytest.mark.parametrize('hours', [80, 104, 128, 152, 176, 200, 224])
@pytest.mark.parametrize('file,field,value', [
    ('training/latest.json', 'state', 'state_004159'),
    ('training/latest.json', 'iteration', 4159),
    ('training/manifest.json', 'source_fingerprint', 'wrong'),
    ('training/manifest.json', 'seed', 123),
    ('training/manifest.json', 'workers', 1),
    ('training/manifest.json', 'smoke', True),
    ('P80.json', 'sha256', 'wrong'),
    ('reload-P80.json', 'status', 'FAILED'),
    ('continuation.json', 'final_recovery_verified', False),
    ('resume.json', 'training_source_archive_sha256', 'wrong'),
    ('resume.json', 'policy_group_sampling', 'uniform'),
])
def test_rejects_incompatible_baseline(file, field, value, hours):
    file = file.replace('P80', f'P{hours}')
    data = metadata(hours)
    assert validate_source_metadata(data.__getitem__, hours)['iteration'] == source_profile(hours)['iteration']
    data[file][field] = value
    with pytest.raises(ValueError):
        validate_source_metadata(data.__getitem__, hours)


@pytest.mark.parametrize('hours', [80, 104, 128, 152, 176, 200, 224])
@pytest.mark.parametrize('damage', [None, 'archive', 'trainer', 'sampler'])
def test_archived_preparation_verifies_without_migration(tmp_path, monkeypatch, damage, hours):
    from hunl_escher import continuation_batch as worker
    p, data = source_profile(hours), metadata(hours)
    source, trainer = tmp_path / 'source', tmp_path / 'trainer'
    source.mkdir(); trainer.mkdir()
    for name, value in data.items():
        path = source / name
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(value))
    archive = source / 'source.tar.gz'
    archive.write_bytes(b'pinned archived trainer')
    if damage == 'sampler':
        data[f'P{hours}.json']['metrics']['policy_validation']['group_sampling'] = 'uniform'
        (source / f'P{hours}.json').write_text(json.dumps(data[f'P{hours}.json']))
    before = {str(f.relative_to(source)): f.read_bytes() for f in source.rglob('*') if f.is_file()}
    monkeypatch.setattr(worker, 'storage', lambda *args: None)
    monkeypatch.setattr(worker, 'sha256_file', lambda path: 'wrong' if damage == 'archive' else p['archive_sha256'])
    monkeypatch.setattr(worker, 'fingerprint', lambda path: 'wrong' if damage == 'trainer' else p['fingerprint'])
    if damage:
        with pytest.raises(ValueError):
            worker.prepare_archived_source(source, trainer, tmp_path, hours)
        assert not (source / 'training_source.tar.gz').exists()
    else:
        assert worker.prepare_archived_source(source, trainer, tmp_path, hours) == (source, trainer)
        assert (source / 'training_source.tar.gz').read_bytes() == archive.read_bytes()
        assert json.loads((tmp_path / 'source-verification.json').read_text())['migration_required'] is False
    assert all((source / name).read_bytes() == value for name, value in before.items())
