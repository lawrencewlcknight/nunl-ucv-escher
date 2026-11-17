"""Finish publication/evaluation of completed P32 training; never train again."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .checkpointing import sha256_file
from .continuation import LABELS, checked_pointer
from .continuation_cloud import publisher
from .continuation_evaluate import run_suite
from .experiment import upload_checkpoint
from .storage_transfer import run_transfer
from .train import write_json


def validate_completed_training(root):
    root = Path(root)
    report = json.loads((root / 'continuation.json').read_text())
    pointer = checked_pointer(root / 'training')
    if report.get('smoke') or set(report['checkpoints']) != set(LABELS):
        raise ValueError('All five production checkpoints are required')
    final = report['checkpoints']['P32']
    if pointer['state'] != final['recovery'] or pointer['iteration'] != final['outer_iteration']:
        raise ValueError('Final recovery pointer differs from P32')
    for label in LABELS:
        marker = report['checkpoints'][label]
        if label != 'P8':
            stage = report['stages'][label]
            if stage['returncode'] != 0 or stage['timed_out'] or marker['active_seconds'] < marker['target_active_seconds']:
                raise ValueError('A training stage did not complete')
        if sha256_file(root / f'{label}.pkl') != marker['sha256']:
            raise ValueError(f'{label} policy hash differs from the recorded export')
        on_disk = json.loads((root / f'{label}.json').read_text())
        if any(on_disk[k] != marker[k] for k in ('sha256', 'outer_iteration', 'active_seconds')):
            raise ValueError(f'{label} publication marker differs')
    return report, pointer


def verify(source, training, policy, report):
    subprocess.run([sys.executable, str(Path(__file__).with_name('verify_recovery.py')),
        '--source', str(source), '--training', str(training), '--policy', str(policy),
        '--report', str(report)], check=True, timeout=1800,
        cwd=source, env={**os.environ, 'PYTHONPATH': str(source), 'PYTHONNOUSERSITE': '1'})
    if json.loads(report.read_text())['status'] != 'VERIFIED':
        raise ValueError('Recovery verification did not succeed')


def recover(workspace, uri, *, evaluation_seconds=14400):
    workspace = Path(workspace).resolve()
    root, source = workspace / 'run', workspace / 'archived-trainer'
    original, pointer = validate_completed_training(root)
    run_transfer(['gcloud', 'storage', 'cp', uri + '/launch.json', str(root / 'launch.json')], timeout=120)
    backup = root / 'continuation-before-recovery.json'
    if not backup.exists():
        shutil.copyfile(root / 'continuation.json', backup)
    audit = root / 'recovery-audit.json'
    with publisher(root, uri, training=True) as sync:
        try:
            write_json(audit, dict(status='VERIFYING_DISK', state=pointer['state']))
            sync()
            verify(source, root / 'training', root / 'P32.pkl', root / 'reload-P32-disk.json')
            # Retry publication without pruning anything yet. Then download the
            # cloud copy into a separate directory and verify SHA256 + full restore.
            run_transfer(['gcloud', 'storage', 'rsync', '--recursive',
                str(root / 'training' / pointer['state']), uri + '/training/' + pointer['state']], timeout=1800)
            cloud = workspace / 'cloud-verification'
            cloud.mkdir(exist_ok=True)
            for name in ('manifest.json', 'latest.json'):
                shutil.copyfile(root / 'training' / name, cloud / name)
            write_json(audit, dict(status='VERIFYING_CLOUD', state=pointer['state']))
            sync()
            run_transfer(['gcloud', 'storage', 'rsync', '--recursive',
                uri + '/training/' + pointer['state'], str(cloud / pointer['state'])], timeout=1800)
            verify(source, cloud, root / 'P32.pkl', root / 'reload-P32.json')
            # Both disk and cloud copies are verified before policy publication
            # and pruning old continuation recoveries. Original run is untouched.
            upload_checkpoint(root, uri, 'P32')
            original.update(status='TRAINING_COMPLETE', final_recovery_verified=True,
                            recovered_after_publication_failure=True)
            original.pop('error_type', None)
            original.pop('reason', None)
            write_json(root / 'continuation.json', original)
            write_json(audit, dict(status='EVALUATING', state=pointer['state'],
                disk_verified=True, cloud_verified=True, policy_sha256=original['checkpoints']['P32']['sha256']))
            sync(final=True)
            evaluation = run_suite(root, root / 'evaluation', workers=8, max_seconds=evaluation_seconds)
            sync(final=True)
            plan = json.loads((root / 'launch.json').read_text())
            write_json(root / 'slumbot_ready.json', dict(status='READY', policy='P32',
                sha256=original['checkpoints']['P32']['sha256'], source_uri=plan['source_uri'],
                local_evaluation_status=evaluation['status']))
            write_json(audit, dict(status='COMPLETE', state=pointer['state'], disk_verified=True,
                cloud_verified=True, local_evaluation_status=evaluation['status'],
                policy_sha256=original['checkpoints']['P32']['sha256']))
        except Exception as exc:
            write_json(audit, dict(status='FAILED', error_type=type(exc).__name__, reason=str(exc)))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--uri', required=True)
    parser.add_argument('--evaluation-seconds', type=float, default=14400)
    args = parser.parse_args()
    recover(args.workspace, args.uri.rstrip('/'), evaluation_seconds=args.evaluation_seconds)
