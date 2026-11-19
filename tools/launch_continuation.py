#!/usr/bin/env python3
"""Prepare a bounded continuation VM; --execute submits the prepared launch."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess

from launch_learning_check import source_archive


def build_bundle(output, *, project, zone, bucket, service_account, run_name,
                 source_uri, expected_p8_sha256, baseline_hours=8):
    if baseline_hours < 8 or (baseline_hours-8) % 24:
        raise ValueError('Invalid baseline hours')
    labels = [f'P{baseline_hours+6*i}' for i in range(5)]
    if not re.fullmatch(r'[a-z][a-z0-9-]{0,45}[a-z0-9]',run_name):
        raise ValueError('Invalid run name')
    for uri in (bucket,source_uri):
        if not re.fullmatch(r'gs://[a-z0-9][a-z0-9._-]+(?:/[a-zA-Z0-9_./-]+)?',uri):
            raise ValueError('Invalid Cloud Storage URI')
    if not re.fullmatch(r'[a-f0-9]{64}',expected_p8_sha256):
        raise ValueError('Expected baseline SHA-256 is required')
    output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    uri=bucket.rstrip('/')+'/'+run_name
    source_uri=source_uri.rstrip('/')
    if uri==source_uri or uri.startswith(source_uri+'/') or source_uri.startswith(uri+'/'):
        raise ValueError('Use a separate cloud prefix for the continuation')
    source_archive(Path(__file__).resolve().parents[1],output/'source.tar.gz')
    digest=hashlib.sha256((output/'source.tar.gz').read_bytes()).hexdigest()
    command=['/opt/hunl/venv/bin/python','-m','hunl_escher.continuation_cloud',
        '--workspace','/var/lib/hunl-continuation','--source-uri',source_uri,'--output-uri',uri,
        '--expected-p8-sha256',expected_p8_sha256, '--baseline-hours', str(baseline_hours)]
    script=output/'startup.sh'
    script.write_text('\n'.join(['#!/bin/bash','set -euo pipefail',
        'export DEBIAN_FRONTEND=noninteractive',
        'if [ -e /var/lib/hunl-continuation-started ]; then shutdown -h now; exit 0; fi',
        'touch /var/lib/hunl-continuation-started',"trap 'shutdown -h now' EXIT",
        'apt-get update -qq','apt-get install -y -qq python3-venv google-cloud-cli',
        'mkdir -p /opt/hunl/coordinator','cd /opt/hunl/coordinator',
        shlex.join(['gcloud','storage','cp',uri+'/source.tar.gz','/opt/hunl/coordinator.tar.gz']),
        f"echo '{digest}  /opt/hunl/coordinator.tar.gz' | sha256sum --check",
        'tar -xzf /opt/hunl/coordinator.tar.gz -C /opt/hunl/coordinator',
        'python3 -m venv /opt/hunl/venv',
        '/opt/hunl/venv/bin/python -m pip install --upgrade pip',
        '/opt/hunl/venv/bin/python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu',
        "/opt/hunl/venv/bin/python -m pip install '.[parallel,api]'",
        'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1',
        shlex.join(command),'']))
    commands=[['gcloud','storage','cp','--if-generation-match=0',str(output/'source.tar.gz'),uri+'/source.tar.gz'],
        ['gcloud','storage','cp','--if-generation-match=0',str(output/'launch.json'),uri+'/launch.json'],
        ['gcloud','compute','instances','create',run_name+'-train','--project',project,'--zone',zone,
         '--machine-type','c4-highmem-16','--image-project','debian-cloud','--image-family','debian-12',
         '--boot-disk-type','hyperdisk-balanced','--boot-disk-size','200GB',
         '--no-boot-disk-auto-delete',
         '--service-account',service_account,'--scopes','cloud-platform',
         '--metadata-from-file','startup-script='+str(script),'--no-restart-on-failure',
         '--max-run-duration','36h','--instance-termination-action','STOP',
         '--labels','purpose=hunl-continuation']]
    plan=dict(uri=uri,source_uri=source_uri,baseline_sha256=expected_p8_sha256,
        additional_active_hours=24,stage_hours=6,labels=labels,baseline_hours=baseline_hours,
        machine='c4-highmem-16',collection_workers=8,training_threads=8,
        training_wall_budget_hours=30,evaluation_wall_budget_hours=4,vm_max_hours=36,
        evaluation='Sequentially after training and successful final reload, on the same VM; eight evaluation workers',
        primary=f'{labels[-1]} vs {labels[0]}, alpha 0.05',secondary='Four adjacent head-to-head, alpha 0.01 each',
        recovery='Six-hour wall-clock recovery checks, skipped if saved in the last hour; baseline run preserved; newest continuation recovery retained',
        shutdown='Automatic on exit. Stopped boot disk remains billable until explicitly deleted.',
        source_sha256=digest,commands=commands)
    plan['external'] = dict(machine='c4-standard-4', wall_hours=24, vm_max_hours=26, pilot_hands=1000, confirmation_hands=20000, policy=labels[-1], launch='Explicit second launch after slumbot_ready.json; no waiting VM')
    (output/'launch.json').write_text(json.dumps(plan,indent=2)+'\n')
    return plan


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',required=True)
    p.add_argument('--zone',default='europe-west1-b')
    p.add_argument('--bucket',required=True)
    p.add_argument('--service-account',required=True)
    p.add_argument('--run-name',required=True)
    p.add_argument('--source-uri',required=True)
    p.add_argument('--expected-p8-sha256', '--expected-baseline-sha256', dest='expected_p8_sha256', required=True)
    p.add_argument('--baseline-hours', type=int, default=8)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--execute',action='store_true')
    args=vars(p.parse_args());execute=args.pop('execute')
    plan=build_bundle(**args)
    print(json.dumps(plan,indent=2))
    if execute:
        existing=subprocess.run(['gcloud','storage','ls',plan['uri']+'/**'],capture_output=True,text=True)
        if existing.returncode==0:
            raise RuntimeError('Cloud prefix already exists; choose a new run name')
        for command in plan['commands']:
            subprocess.run(command,check=True)


if __name__=='__main__':
    main()
