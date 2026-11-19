"""Continue a verified full recovery using its unmodified, archived trainer.

The new orchestration never relaxes source/runtime/configuration checks. Each
six-hour subprocess uses the original contract and exports at its final boundary.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time

from .checkpointing import load_checkpoint_payload, sha256_file
from .experiment import publish_policy, upload_checkpoint
from .experiment_runtime import run_child
from .train import write_json

LABELS = ('P8', 'P14', 'P20', 'P26', 'P32')


def continuation_labels(baseline_hours=8):
    if not isinstance(baseline_hours, int) or baseline_hours < 8 or (baseline_hours-8) % 24:
        raise ValueError('Baseline must be P8 plus a whole number of 24-hour continuations')
    return tuple(f'P{baseline_hours + 6*i}' for i in range(5))


def report_labels(report):
    labels = continuation_labels(report.get('baseline_hours', 8))
    if tuple(report.get('labels', labels)) != labels:
        raise ValueError('Continuation labels differ from its baseline')
    return labels


def checked_pointer(training):
    pointer = json.loads((Path(training) / 'latest.json').read_text())
    if not re.fullmatch(r'state_[0-9]{6,}', pointer['state']):
        raise ValueError('Invalid recovery directory')
    if int(pointer['state'].split('_')[1]) != pointer['iteration']:
        raise ValueError('Recovery iteration differs from pointer')
    return pointer


def trainer_command(training, contract, seconds):
    # Scheduling remains identical to the saved contract. Final iteration of
    # each bounded stage triggers policy fitting and full recovery automatically.
    command = [sys.executable, '-m', 'hunl_escher.train', '--resume',
               '--output', str(training), '--iterations', '1000000000',
               '--additional-training-seconds', str(seconds)]
    for key, flag in [('workers', 'workers'), ('threads', 'threads'), ('seed', 'seed'),
                      ('policy_every', 'policy-every'), ('checkpoint_every', 'checkpoint-every'),
                      ('recovery_seconds', 'recovery-seconds'), ('policy_seconds', 'policy-seconds')]:
        command += ['--'+flag, str(contract[key])]
    if contract['smoke']:
        command.append('--smoke')
    else:
        command += ['--config', str(training / 'continuation-config.json')]
    return command


def run_continuation(source_run, trainer_source, output, *, stage_seconds=21600.,
                     wall_seconds=108000., upload_uri=None, allow_smoke=False, baseline_hours=8,
                     verify_each_stage=False):
    labels = continuation_labels(baseline_hours)
    baseline, final = labels[0], labels[-1]
    if not all(math.isfinite(x) and x > 0 for x in (stage_seconds, wall_seconds)):
        raise ValueError('Time budgets must be positive and finite')
    source, trainer, root = map(lambda p: Path(p).resolve(), (source_run, trainer_source, output))
    if root == source or source in root.parents or root in source.parents:
        raise ValueError('Use a separate output directory; preserve the original recovery')
    contract = json.loads((source / 'training/manifest.json').read_text())
    marker = json.loads((source / f'{baseline}.json').read_text())
    pointer = checked_pointer(source / 'training')
    if marker['recovery'] != pointer['state'] or marker['outer_iteration'] != pointer['iteration']:
        raise ValueError(f'{baseline} is not the latest full recovery')
    if sha256_file(source / f'{baseline}.pkl') != marker['sha256']:
        raise ValueError(f'{baseline} policy hash mismatch')
    payload = load_checkpoint_payload(source / f'{baseline}.pkl')
    if payload['outer_iteration'] != pointer['iteration'] or payload['training_elapsed_seconds'] != marker['active_seconds']:
        raise ValueError('Baseline policy and publication marker differ')
    if bool(contract['smoke']) != bool(marker.get('smoke', contract['smoke'])):
        raise ValueError('Inconsistent smoke metadata')
    if contract['smoke'] and not allow_smoke:
        raise ValueError('Production continuation cannot use a smoke checkpoint')
    root.mkdir(parents=True, exist_ok=False)
    report = dict(status='PREPARING', source_run=str(source), trainer_source=str(trainer),
                  training_source_fingerprint=contract['source_fingerprint'],
                  stage_seconds=stage_seconds, additional_active_seconds=4*stage_seconds,
                  smoke=contract['smoke'], baseline_hours=baseline_hours, labels=labels,
                  checkpoints={baseline: marker}, stages={})
    write_json(root / 'continuation.json', report)
    started = time.monotonic()
    deadline = started + wall_seconds
    training = root / 'training'
    # Pin imports for the original trainer even if the coordinator is installed.
    env = {**os.environ, 'PYTHONPATH': str(trainer), 'PYTHONNOUSERSITE': '1'}
    verifier = Path(__file__).with_name('verify_recovery.py')
    def verify(name):
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Continuation wall budget exhausted before reload verification')
        outcome = run_child([sys.executable, str(verifier), '--source', str(trainer),
            '--training', str(training), '--policy', str(root / f'{name}.pkl'),
            '--report', str(root / f'reload-{name}.json')], root / f'reload-{name}.log',
            timeout=min(1800, remaining), cwd=trainer, env=env)
        if outcome['returncode'] != 0:
            raise RuntimeError(f'{name} full-state reload verification failed')
    try:
        training.mkdir()
        shutil.copy2(source / 'training/manifest.json', training / 'manifest.json')
        shutil.copy2(source / 'training/latest.json', training / 'latest.json')
        shutil.copytree(source / 'training' / pointer['state'], training / pointer['state'])
        for suffix in ('pkl', 'json'):
            shutil.copy2(source / f'{baseline}.{suffix}', root / f'{baseline}.{suffix}')
        write_json(training / 'continuation-config.json', contract['config'])
        verify(baseline)  # Source/runtime, every shard checksum, actual solver and actor restore.
        if upload_uri:
            upload_checkpoint(root, upload_uri, baseline)
        previous = marker['active_seconds']
        for index, label in enumerate(labels[1:], 1):
            report.update(status='TRAINING', current_stage=label)
            write_json(root / 'continuation.json', report)
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                raise TimeoutError('Continuation wall budget exhausted')
            # Use absolute cumulative targets, avoiding four accumulated iteration
            # overshoots. Active time excludes extraction, saving and uploads.
            target = marker['active_seconds'] + index*stage_seconds
            if contract['smoke']:
                target = previous + stage_seconds
            duration = target-previous
            if duration <= 0:
                raise RuntimeError('A single iteration crossed multiple checkpoint targets')
            outcome = run_child(trainer_command(training, contract, duration),
                root / f'train-{label}.log', timeout=remaining, cwd=trainer, env=env)
            report['stages'][label] = outcome
            if outcome['returncode'] != 0:
                raise RuntimeError(f'Training {label} failed; see stage log')
            record = publish_policy(training, root, label)
            if record['active_seconds'] < target or record['outer_iteration'] <= report['checkpoints'][labels[index-1]]['outer_iteration']:
                raise RuntimeError('Training stage did not reach its target')
            record['target_active_seconds'] = target
            write_json(root / f'{label}.json', record)
            report['checkpoints'][label] = record
            previous = record['active_seconds']
            if verify_each_stage:
                verify(label)
            if upload_uri:
                upload_checkpoint(root, upload_uri, label)
            write_json(root / 'continuation.json', report)
        report['status'] = 'VERIFYING_FINAL_RECOVERY'
        write_json(root / 'continuation.json', report)
        if not verify_each_stage:
            verify(final)
        report['status'] = 'TRAINING_COMPLETE'
        report['final_recovery_verified'] = True
    except Exception as exc:
        report.update(status='FAILED', error_type=type(exc).__name__, reason=str(exc))
        raise
    finally:
        report['wall_seconds'] = time.monotonic()-started
        write_json(root / 'continuation.json', report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', type=Path, required=True)
    p.add_argument('--trainer-source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stage-seconds', type=float, default=21600)
    p.add_argument('--wall-seconds', type=float, default=108000)
    p.add_argument('--upload-uri')
    p.add_argument('--allow-smoke', action='store_true')
    p.add_argument('--baseline-hours', type=int, default=8)
    run_continuation(**vars(p.parse_args()))


if __name__ == '__main__':
    main()
