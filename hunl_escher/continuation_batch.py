"""Self-contained corrected P56/P80/P104/P128/P152/P176 continuation; no external opponents."""
import argparse
import json
from pathlib import Path
import signal
import shutil

from .checkpointing import sha256_file
from .continuation import run_continuation
from .continuation_cloud import download_source, publisher, storage
from .continuation_evaluate import run_suite
from .corrected_continuation import prepare_corrected_source, fingerprint
from .continuation_sources import source_profile, validate_source_metadata
from .train import write_json

def prepare_archived_source(source, trainer, workspace, baseline_hours):
    """Keep an already corrected trainer and all saved learning state intact."""
    p = source_profile(baseline_hours)
    for name in (f'reload-P{baseline_hours}.json', 'resume.json'):
        storage('cp', p['uri'] + '/' + name, source / name)
    receipt = validate_source_metadata(lambda name: json.loads((source / name).read_text()), baseline_hours)
    if (sha256_file(source / 'source.tar.gz') != p['archive_sha256'] or
            fingerprint(trainer) != p['fingerprint']):
        raise ValueError(f'P{baseline_hours} archived trainer checksum/fingerprint mismatch')
    shutil.copy2(source / 'source.tar.gz', source / 'training_source.tar.gz')
    write_json(workspace / 'source-verification.json', dict(receipt,
        archive_sha256=p['archive_sha256'], migration_required=False,
        policy_group_sampling='objective_mass'))
    # run_continuation strictly reloads the full state before any training.
    return source, trainer


def run_worker(workspace, output_uri, baseline_hours=56):
    profile = source_profile(baseline_hours)
    source_uri = profile['uri']
    uri = output_uri.rstrip('/')
    if uri == source_uri or uri.startswith(source_uri + '/') or source_uri.startswith(uri + '/'):
        raise ValueError('Use a separate continuation output prefix')
    workspace = Path(workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=False)
    # Status exists even if download or migration fails. The run directory itself
    # must remain absent until run_continuation creates it.
    output = workspace / 'run'
    status_root = workspace / 'status'
    status_root.mkdir()
    status = dict(status='PREPARING', source_uri=source_uri, output_uri=uri,
                  baseline_hours=baseline_hours, baseline_sha256=profile['policy_sha256'], policy_group_sampling='objective_mass',
                  external_benchmarks=False)
    def stage(name, **fields):
        status.update(status=name, **fields)
        write_json(status_root / 'batch_status.json', status)
        print(json.dumps(status), flush=True)
    stage('DOWNLOADING')
    with publisher(status_root, uri), publisher(output, uri, training=True) as sync:
        try:
            source, trainer = download_source(source_uri, workspace / 'input', profile['policy_sha256'], baseline_hours)
            if baseline_hours == 56:
                stage('VERIFYING_AND_MIGRATING')
                source, trainer = prepare_corrected_source(source, trainer, workspace)
                receipts = ('sampler-migration.json', 'original-P56-verified.json', 'original-P56-verify.log')
            else:
                stage('VERIFYING_CORRECTED_SOURCE')
                source, trainer = prepare_archived_source(source, trainer, workspace, baseline_hours)
                receipts = ('source-verification.json',)
            for name in receipts:
                storage('cp', workspace / name, uri + '/' + name)
            archive = source / 'training_source.tar.gz'
            storage('cp', '--if-generation-match=0', archive, uri + '/training_source.tar.gz')
            stage('TRAINING')
            report = run_continuation(source, trainer, output, baseline_hours=baseline_hours,
                                      stage_seconds=21600, wall_seconds=108000, upload_uri=uri,
                                      verify_each_stage=True)
            # Fail closed if the trained artifacts do not attest to the fix.
            for label in report['labels'][1:]:
                marker = report['checkpoints'][label]
                diagnostics = marker['metrics']['policy_validation']
                if (diagnostics.get('group_sampling') != 'objective_mass' or
                        diagnostics.get('policy_sampling_seed') != [profile['seed'], marker['outer_iteration']]):
                    raise ValueError('Policy fit did not use the audited corrected sampler')
            write_json(output / 'resume.json', dict(
                training_source_archive=uri + '/training_source.tar.gz',
                training_source_archive_sha256=sha256_file(archive),
                training_manifest=uri + '/training/manifest.json',
                latest_pointer=uri + '/training/latest.json',
                source_fingerprint=report['training_source_fingerprint'],
                policy_group_sampling='objective_mass',
                instructions='Restore full state with this patched archived trainer and exact runtime; '
                             'a playable policy alone cannot resume training. No automatic retries.'))
            sync()
            stage('EVALUATING', final_policy_sha256=report['checkpoints'][f'P{baseline_hours + 24}']['sha256'])
            evaluation = run_suite(output, output / 'evaluation', workers=8,
                                   max_seconds=14400, external_benchmarks=False)
            sync(final=True)
            stage(evaluation['status'], final_recovery_verified=True,
                  evaluation_status=evaluation['status'])
        except BaseException as exc:
            stage('INTERRUPTED' if isinstance(exc, KeyboardInterrupt) else 'FAILED',
                  error_type=type(exc).__name__, reason=str(exc))
            raise
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output-uri', required=True)
    parser.add_argument('--baseline-hours', type=int, choices=(56, 80, 104, 128, 152, 176), default=56)
    args = vars(parser.parse_args())
    def terminate(signum, frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise KeyboardInterrupt('Batch task terminated')
    signal.signal(signal.SIGTERM, terminate)
    if run_worker(**args)['status'] != 'COMPLETE':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
