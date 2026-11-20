#!/usr/bin/env python3
"""Prepare separate P56/P80 strategic-position Batch audits; --execute submits."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hunl_escher.position_audit_config import (
    AUDIT_SECONDS, DEVELOPMENT_POSITIONS, LIBRARY_SHA256, frozen_library, policies)
from batch_runtime import bootstrap, job_spec
from launch_learning_check import source_archive


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_bundle(output, *, checkpoint_hours, run_name, project='clever-overview-399515',
                 region='europe-west1', bucket='gs://clever-overview-399515-fhp-escher-results/hunl',
                 service_account='fhp-escher-runner@clever-overview-399515.iam.gserviceaccount.com'):
    specs = policies(checkpoint_hours)
    frozen_library()  # Never regenerate positions using a changed generator.
    if not re.fullmatch(r'[a-z][a-z0-9-]{1,61}[a-z0-9]', run_name):
        raise ValueError('Invalid Batch job name')
    if not re.fullmatch(r'[a-z]+-[a-z]+[0-9]+', region):
        raise ValueError('Invalid region')
    if not re.fullmatch(r'gs://[a-z0-9][a-z0-9._-]+(?:/[a-zA-Z0-9_./-]+)?', bucket):
        raise ValueError('Invalid bucket prefix')
    uri = bucket.rstrip('/') + '/' + run_name
    if any(uri == s['uri'] or uri.startswith(s['uri'] + '/') or s['uri'].startswith(uri + '/') for s in specs):
        raise ValueError('Use an independent output prefix')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_archive(Path(__file__).resolve().parents[1], output / 'source.tar.gz')
    digest = sha256(output / 'source.tar.gz')
    command = ['timeout', '--signal=TERM', '--kill-after=120s', '1200s',
               'python', '-u', '-m', 'hunl_escher.position_audit_batch',
               '--workspace', '/workspace/position-audit', '--output-uri', uri,
               '--checkpoint-hours', str(checkpoint_hours)]
    script = bootstrap(command, uri, digest, parallel=False)
    job = job_spec(script, service_account, boot_disk_gb=30, max_run_seconds=3600,
                   labels=dict(experiment=f'hunl-positions-p{checkpoint_hours}', purpose='strategic-audit'))
    job['allocationPolicy']['location'] = dict(allowedLocations=['regions/' + region])
    job['allocationPolicy']['instances'][0]['policy']['machineType'] = 'c4-standard-2'
    job['taskGroups'][0]['taskSpec']['computeResource'] = dict(cpuMilli=2000, memoryMib=6144)
    (output / 'job.json').write_text(json.dumps(job, indent=2) + '\n')
    (output / 'batch.sh').write_text(script)
    plan = dict(experiment=f'hunl_positions_p{checkpoint_hours}_v1', backend='google_cloud_batch',
        checkpoint_hours=checkpoint_hours, job_name=run_name, project=project, region=region, uri=uri,
        policies=specs, source_sha256=digest, job_sha256=sha256(output / 'job.json'),
        library_sha256=LIBRARY_SHA256, partition='development', assessment_unlocked=False,
        positions_per_policy=DEVELOPMENT_POSITIONS, reference='P32 re-scored in the same runtime',
        machine='c4-standard-2', boot_disk_gb=30, threads=1, parallel_workers=0,
        audit_seconds=AUDIT_SECONDS, worker_seconds=1200, max_job_seconds=3600, automatic_retries=0,
        training=False, external_benchmarks=False,
        outputs='audit/report.json, audit/report.html, per-position records, provenance and frozen library',
        interpretation='Conditional bb per audited decision; no whole-game win rate or population significance claim',
        cleanup='Batch releases worker VM and disk; analytical reports remain in GCS')
    (output / 'launch.json').write_text(json.dumps(plan, indent=2) + '\n')
    return plan


def submit(output):
    output = Path(output)
    plan = json.loads((output / 'launch.json').read_text())
    if (sha256(output / 'source.tar.gz') != plan['source_sha256'] or
            sha256(output / 'job.json') != plan['job_sha256'] or
            plan['policies'] != policies(plan['checkpoint_hours']) or
            plan['library_sha256'] != LIBRARY_SHA256):
        raise ValueError('Prepared bundle identity/checksum mismatch')
    existing = subprocess.run(['gcloud', 'batch', 'jobs', 'list', '--project', plan['project'],
        '--location', plan['region'], '--format=json'], check=True, capture_output=True, text=True, timeout=120)
    if any(j['name'].rsplit('/', 1)[-1] == plan['job_name'] for j in json.loads(existing.stdout)):
        raise ValueError('Batch job already exists; inspect it instead of submitting again')
    receipts = {}
    for spec in plan['policies']:
        result = subprocess.run(['gcloud', 'storage', 'cat', spec['uri'] + '/' + spec['label'] + '.json'],
            check=True, capture_output=True, text=True, timeout=120)
        marker = json.loads(result.stdout)
        if marker['sha256'] != spec['policy_sha256'] or marker['outer_iteration'] != spec['iteration']:
            raise ValueError('Published policy does not match the pinned audit source')
        receipts[spec['label']] = marker
    (output / 'source-verification.json').write_text(json.dumps(receipts, indent=2) + '\n')
    for name in ('launch.json', 'source.tar.gz', 'job.json', 'source-verification.json'):
        subprocess.run(['gcloud', 'storage', 'cp', '--if-generation-match=0', str(output / name),
                        plan['uri'] + '/' + name], check=True, timeout=300)
    result = subprocess.run(['gcloud', 'batch', 'jobs', 'submit', plan['job_name'],
        '--project', plan['project'], '--location', plan['region'], '--config', str(output / 'job.json'),
        '--format=json'], check=True, capture_output=True, text=True, timeout=180)
    (output / 'submission.json').write_text(result.stdout)
    print(result.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-hours', type=int, choices=(56, 80))
    parser.add_argument('--run-name')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--submit-prepared', type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.submit_prepared:
        if args.checkpoint_hours or args.run_name or args.output or args.execute:
            parser.error('--submit-prepared cannot be combined with preparation options')
        submit(args.submit_prepared)
    else:
        if args.checkpoint_hours is None or not args.run_name or not args.output:
            parser.error('--checkpoint-hours, --run-name and --output are required')
        plan = build_bundle(args.output, checkpoint_hours=args.checkpoint_hours, run_name=args.run_name)
        print(json.dumps(plan, indent=2))
        if args.execute:
            submit(args.output)


if __name__ == '__main__':
    main()
