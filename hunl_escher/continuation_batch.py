"""Self-contained corrected P56-to-P80 Batch worker; no external opponents."""
import argparse
import json
from pathlib import Path
import signal

from .checkpointing import sha256_file
from .continuation import run_continuation
from .continuation_cloud import download_source, publisher, storage
from .continuation_evaluate import run_suite
from .corrected_continuation import prepare_corrected_source, P56_SHA256
from .train import write_json

SOURCE_URI = 'gs://clever-overview-399515-fhp-escher-results/hunl/hunl-p32-plus24-20261117-121800'


def run_worker(workspace, output_uri):
    uri = output_uri.rstrip('/')
    if uri == SOURCE_URI or uri.startswith(SOURCE_URI + '/') or SOURCE_URI.startswith(uri + '/'):
        raise ValueError('Use a separate continuation output prefix')
    workspace = Path(workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=False)
    # Status exists even if download or migration fails. The run directory itself
    # must remain absent until run_continuation creates it.
    output = workspace / 'run'
    status_root = workspace / 'status'
    status_root.mkdir()
    status = dict(status='PREPARING', source_uri=SOURCE_URI, output_uri=uri,
                  baseline_sha256=P56_SHA256, policy_group_sampling='objective_mass',
                  external_benchmarks=False)
    def stage(name, **fields):
        status.update(status=name, **fields)
        write_json(status_root / 'batch_status.json', status)
        print(json.dumps(status), flush=True)
    stage('DOWNLOADING')
    with publisher(status_root, uri), publisher(output, uri, training=True) as sync:
        try:
            source, trainer = download_source(SOURCE_URI, workspace / 'input', P56_SHA256, 56)
            stage('VERIFYING_AND_MIGRATING')
            source, trainer = prepare_corrected_source(source, trainer, workspace)
            for name in ('sampler-migration.json', 'original-P56-verified.json', 'original-P56-verify.log'):
                storage('cp', workspace / name, uri + '/' + name)
            archive = source / 'training_source.tar.gz'
            storage('cp', '--if-generation-match=0', archive, uri + '/training_source.tar.gz')
            stage('TRAINING')
            report = run_continuation(source, trainer, output, baseline_hours=56,
                                      stage_seconds=21600, wall_seconds=108000, upload_uri=uri,
                                      verify_each_stage=True)
            # Fail closed if the trained artifacts do not attest to the fix.
            for label in report['labels'][1:]:
                marker = report['checkpoints'][label]
                diagnostics = marker['metrics']['policy_validation']
                if (diagnostics.get('group_sampling') != 'objective_mass' or
                        diagnostics.get('policy_sampling_seed') != [20261115, marker['outer_iteration']]):
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
            stage('EVALUATING', final_policy_sha256=report['checkpoints']['P80']['sha256'])
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
    args = vars(parser.parse_args())
    def terminate(signum, frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise KeyboardInterrupt('Batch task terminated')
    signal.signal(signal.SIGTERM, terminate)
    if run_worker(**args)['status'] != 'COMPLETE':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
