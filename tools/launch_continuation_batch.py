#!/usr/bin/env python3
"""Prepare/submit one corrected P56+24 Batch job with local comparisons only."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from launch_learning_check import source_archive
from batch_runtime import PYTHON_IMAGE, VM_IMAGE, bootstrap, job_spec

SOURCE_URI = 'gs://clever-overview-399515-fhp-escher-results/hunl/hunl-p32-plus24-20261117-121800'
BASELINE_SHA256 = 'a1ec410adc7aab98c07045a78ea37b42e7a4c7c3a6bb648afa7545a18d4ce94a'
TRAINER_FINGERPRINT = 'a3b341129c3b2af43b93f9799482169393bd07b0dc79692e10a5d631ad045809'


def build_bundle(output, *, run_name, project='clever-overview-399515', region='europe-west1',
                 bucket='gs://clever-overview-399515-fhp-escher-results/hunl',
                 service_account='fhp-escher-runner@clever-overview-399515.iam.gserviceaccount.com'):
    if not re.fullmatch(r'[a-z][a-z0-9-]{1,61}[a-z0-9]', run_name):
        raise ValueError('Invalid Batch job name')
    if not re.fullmatch(r'[a-z]+-[a-z]+[0-9]+', region):
        raise ValueError('Invalid Batch region')
    if not re.fullmatch(r'gs://[a-z0-9][a-z0-9._-]+(?:/[a-zA-Z0-9_./-]+)?', bucket):
        raise ValueError('Invalid bucket prefix')
    uri = bucket.rstrip('/') + '/' + run_name
    if uri == SOURCE_URI or uri.startswith(SOURCE_URI + '/') or SOURCE_URI.startswith(uri + '/'):
        raise ValueError('Use an independent output prefix')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_archive(Path(__file__).resolve().parents[1], output / 'source.tar.gz')
    digest = hashlib.sha256((output / 'source.tar.gz').read_bytes()).hexdigest()
    command = ['timeout', '--signal=TERM', '--kill-after=300s', '126000s',
               'python', '-u', '-m', 'hunl_escher.continuation_batch',
               '--workspace', '/workspace/continuation', '--output-uri', uri]
    script = bootstrap(command, uri, digest)
    job = job_spec(script, service_account, boot_disk_gb=200, max_run_seconds=129600,
                   labels=dict(experiment='hunl-p56-plus24', sampler='objective-mass'))
    job['allocationPolicy']['location'] = dict(allowedLocations=['regions/' + region])
    plan = dict(experiment='hunl_p56_plus24_objective_mass_v1', backend='google_cloud_batch',
        job_name=run_name, project=project, region=region, uri=uri, source_uri=SOURCE_URI,
        source_sha256=digest, baseline_sha256=BASELINE_SHA256,
        original_trainer_fingerprint=TRAINER_FINGERPRINT,
        machine='c4-highmem-16', boot_disk_gb=200, collection_workers=8, training_threads=8,
        labels=['P56', 'P62', 'P68', 'P74', 'P80'], additional_active_hours=24, stage_hours=6,
        policy_group_sampling='objective_mass', python_container=PYTHON_IMAGE, vm_image=VM_IMAGE,
        training_wall_hours=30, evaluation_wall_hours=4, worker_wall_hours=35, max_job_hours=36,
        automatic_retries=0, external_benchmarks=False,
        primary='P80 vs original P56, alpha 0.05', secondary='Four adjacent comparisons, alpha 0.01 each',
        evaluation='After training; eight workers; pilot excluded; matched deals with exchanged seats',
        recovery='Every stage; keep all playable policies and newest full recovery; preserve prior runs',
        cleanup='Batch releases its VM and disk; results, migration receipt and full recovery remain in GCS')
    for name, value in (('job.json', job), ('launch.json', plan)):
        (output / name).write_text(json.dumps(value, indent=2) + '\n')
    (output / 'batch.sh').write_text(script)
    return plan


def verify_source():
    def read(name):
        result = subprocess.run(['gcloud', 'storage', 'cat', SOURCE_URI + '/' + name],
                                check=True, capture_output=True, text=True, timeout=120)
        return json.loads(result.stdout)
    report, policy, pointer, contract, reload = [read(name) for name in (
        'continuation.json', 'P56.json', 'training/latest.json', 'training/manifest.json', 'reload-P56.json')]
    if (report['status'] != 'TRAINING_COMPLETE' or not report.get('final_recovery_verified') or
            report['labels'][-1] != 'P56' or report['checkpoints']['P56']['sha256'] != BASELINE_SHA256 or
            policy['sha256'] != BASELINE_SHA256 or policy['outer_iteration'] != 2941 or
            policy['recovery'] != 'state_002941' or pointer['state'] != 'state_002941' or
            pointer['iteration'] != 2941 or contract['source_fingerprint'] != TRAINER_FINGERPRINT or
            contract['workers'] != 8 or contract['threads'] != 8 or contract['smoke'] or
            reload['status'] != 'VERIFIED' or reload['iteration'] != 2941 or
            reload['source_fingerprint'] != TRAINER_FINGERPRINT):
        raise ValueError('P56 source identity/recovery is not the approved baseline')
    return dict(status='VERIFIED_METADATA', baseline_sha256=BASELINE_SHA256, iteration=2941,
                full_reload='Repeated on Batch before migration and before training')


def submit(output, plan):
    output = Path(output)
    if hashlib.sha256((output / 'source.tar.gz').read_bytes()).hexdigest() != plan['source_sha256']:
        raise ValueError('Prepared source archive checksum mismatch')
    # A successful list establishes access; an auth/network failure is never
    # interpreted as evidence that the job name is unused.
    existing = subprocess.run(['gcloud', 'batch', 'jobs', 'list', '--project', plan['project'],
        '--location', plan['region'], '--format=json'], check=True, capture_output=True, text=True, timeout=120)
    if any(x['name'].rsplit('/', 1)[-1] == plan['job_name'] for x in json.loads(existing.stdout)):
        raise ValueError('Batch job already exists; do not submit a duplicate')
    receipt = verify_source()
    (output / 'source-verification.json').write_text(json.dumps(receipt, indent=2) + '\n')
    # Reserve the prefix first. A partial submission requires inspection, never
    # an automatic re-run or replacement of existing launch artifacts.
    for name in ('launch.json', 'source.tar.gz', 'job.json', 'source-verification.json'):
        subprocess.run(['gcloud', 'storage', 'cp', '--if-generation-match=0',
            str(output / name), plan['uri'] + '/' + name], check=True, timeout=300)
    result = subprocess.run(['gcloud', 'batch', 'jobs', 'submit', plan['job_name'],
        '--project', plan['project'], '--location', plan['region'],
        '--config', str(output / 'job.json'), '--format=json'], check=True,
        capture_output=True, text=True, timeout=180)
    (output / 'submission.json').write_text(result.stdout)
    print(result.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--submit-prepared', type=Path,
                        help='Submit an already reviewed bundle without rebuilding it')
    parser.add_argument('--project', default='clever-overview-399515')
    parser.add_argument('--region', default='europe-west1')
    parser.add_argument('--bucket', default='gs://clever-overview-399515-fhp-escher-results/hunl')
    parser.add_argument('--service-account', default='fhp-escher-runner@clever-overview-399515.iam.gserviceaccount.com')
    parser.add_argument('--execute', action='store_true')
    args = vars(parser.parse_args()); execute = args.pop('execute')
    prepared = args.pop('submit_prepared')
    if prepared:
        if args['run_name'] or args['output'] or execute:
            parser.error('--submit-prepared cannot be combined with --run-name, --output or --execute')
        submit(prepared, json.loads((prepared / 'launch.json').read_text()))
        return
    if not args['run_name'] or not args['output']:
        parser.error('--run-name and --output are required when preparing a new bundle')
    plan = build_bundle(**args)
    print(json.dumps(plan, indent=2))
    if execute:
        submit(args['output'], plan)


if __name__ == '__main__':
    main()
