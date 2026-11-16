#!/usr/bin/env python3
"""Launch the small Slumbot VM only after the continuation has published P32 readiness."""
import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess


def read_remote(uri):
    result = subprocess.run(['gcloud', 'storage', 'cat', uri], check=True,
                            capture_output=True, text=True, timeout=60)
    return json.loads(result.stdout)


def validate_ready(plan, ready, training, marker):
    if (ready.get('status') != 'READY' or ready.get('policy') != 'P32' or
        ready.get('source_uri') != plan['source_uri'] or
        training.get('status') != 'TRAINING_COMPLETE' or
        not training.get('final_recovery_verified') or training.get('smoke', True)):
        raise ValueError('Continuation is not ready for production Slumbot evaluation')
    expected = training['checkpoints']['P32']['sha256']
    if not re.fullmatch('[a-f0-9]{64}', expected) or ready['sha256'] != expected or marker['sha256'] != expected:
        raise ValueError('P32 publication identities do not agree')
    if training['checkpoints']['P8']['sha256'] != plan['baseline_sha256']:
        raise ValueError('Continuation baseline differs from the launch plan')
    return expected


def build_bundle(plan, output, expected_p32_sha256):
    """Use the source archive already deployed for training, not current local files."""
    output = Path(output).resolve()
    uri = plan['uri'].rstrip('/')
    command = next(c for c in plan['commands'] if c[1:4] == ['compute', 'instances', 'create'])
    def option(name):
        return command[command.index(name)+1]
    name = command[4].removesuffix('-train') + '-slumbot'
    output.mkdir(parents=True, exist_ok=False)
    script = output/'startup.sh'
    worker = ['/opt/hunl/venv/bin/python', '-m', 'hunl_escher.continuation_cloud',
        '--role', 'slumbot', '--workspace', '/var/lib/hunl-slumbot', '--output-uri', uri,
        '--expected-p32-sha256', expected_p32_sha256]
    digest = plan['source_sha256']
    if not re.fullmatch('[a-f0-9]{64}', digest) or not re.fullmatch('[a-f0-9]{64}', expected_p32_sha256):
        raise ValueError('Invalid archive or policy digest')
    script.write_text('\n'.join(['#!/bin/bash', 'set -euo pipefail',
        'export DEBIAN_FRONTEND=noninteractive',
        'if [ -e /var/lib/hunl-slumbot-started ]; then shutdown -h now; exit 0; fi',
        'touch /var/lib/hunl-slumbot-started', "trap 'shutdown -h now' EXIT",
        'apt-get update -qq', 'apt-get install -y -qq python3-venv google-cloud-cli',
        'mkdir -p /opt/hunl/source', 'cd /opt/hunl/source',
        shlex.join(['gcloud', 'storage', 'cp', uri+'/source.tar.gz', '/opt/hunl/source.tar.gz']),
        f"echo '{digest}  /opt/hunl/source.tar.gz' | sha256sum --check",
        'tar -xzf /opt/hunl/source.tar.gz -C /opt/hunl/source',
        'python3 -m venv /opt/hunl/venv',
        '/opt/hunl/venv/bin/python -m pip install --upgrade pip',
        '/opt/hunl/venv/bin/python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu',
        "/opt/hunl/venv/bin/python -m pip install '.[parallel,api]'",
        'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1',
        shlex.join(worker), '']))
    launch = dict(uri=uri, policy_sha256=expected_p32_sha256, source_sha256=digest,
        commands=[['gcloud', 'storage', 'cp', '--if-generation-match=0', str(output/'launch.json'), uri+'/slumbot-launch.json'],
        ['gcloud', 'compute', 'instances', 'create', name,
         '--project', option('--project'), '--zone', option('--zone'),
         '--machine-type', 'c4-standard-4', '--image-project', 'debian-cloud', '--image-family', 'debian-12',
         '--boot-disk-type', 'hyperdisk-balanced', '--boot-disk-size', '30GB',
         '--service-account', option('--service-account'), '--scopes', 'cloud-platform',
         '--metadata-from-file', 'startup-script='+str(script), '--no-restart-on-failure',
         '--max-run-duration', '26h', '--instance-termination-action', 'STOP',
         '--labels', 'purpose=hunl-continuation-slumbot']])
    (output/'launch.json').write_text(json.dumps(launch, indent=2)+'\n')
    return launch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    plan = json.loads(args.training_plan.read_text())
    uri = plan['uri'].rstrip('/')
    # Missing/unready markers fail before creating any VM or reserving a launch.
    expected = validate_ready(plan, *[read_remote(uri+'/'+name) for name in
        ('slumbot_ready.json', 'continuation.json', 'P32.json')])
    launch = build_bundle(plan, args.output, expected)
    print(json.dumps(launch, indent=2))
    if args.execute:
        for command in launch['commands']:
            subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
