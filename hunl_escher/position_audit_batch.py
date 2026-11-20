"""Inference-only P56/P80 audit and P32 comparison; no training or external API."""
import argparse
import json
from pathlib import Path
import signal
import subprocess

from .checkpointing import load_checkpoint_payload, sha256_file
from .position_audit_config import (AUDIT_SECONDS, DEVELOPMENT_POSITIONS,
                                    LIBRARY_SHA256, frozen_library, policies)
from .strategic_audit.workflow import run
from .train import write_json


def storage(*args, timeout=120):
    subprocess.run(['gcloud', 'storage', *map(str, args)], check=True, timeout=timeout)


def fetch_policy(spec, destination):
    label = spec['label']
    marker_path, policy_path = destination / f'{label}.json', destination / f'{label}.pkl'
    storage('cp', spec['uri'] + f'/{label}.json', marker_path)
    marker = json.loads(marker_path.read_text())
    if (marker['sha256'] != spec['policy_sha256'] or
            marker['outer_iteration'] != spec['iteration']):
        raise ValueError(f'{label} publication identity mismatch')
    storage('cp', spec['uri'] + f'/{label}.pkl', policy_path)
    # Verify trusted bytes before unpickling; never load a full recovery.
    if sha256_file(policy_path) != spec['policy_sha256']:
        raise ValueError(f'{label} policy checksum mismatch')
    payload = load_checkpoint_payload(policy_path)
    if (payload['outer_iteration'] != spec['iteration'] or
            payload['training_elapsed_seconds'] != marker['active_seconds']):
        raise ValueError(f'{label} policy and publication marker differ')
    return policy_path, marker


def publish(results, uri):
    # Publish status last, so COMPLETE is only visible after all analytical files.
    storage('rsync', '--recursive', '--exclude', r'(^|/)batch_status\.json$|.*\.tmp$|.*\.lock$',
            results, uri, timeout=300)
    storage('cp', results / 'batch_status.json', uri + '/batch_status.json')


def run_worker(workspace, output_uri, checkpoint_hours):
    specs = policies(checkpoint_hours)
    uri = output_uri.rstrip('/')
    if any(uri == s['uri'] or uri.startswith(s['uri'] + '/') or s['uri'].startswith(uri + '/') for s in specs):
        raise ValueError('Audit output must be separate from every policy source')
    root = Path(workspace).resolve()
    root.mkdir(parents=True, exist_ok=False)
    inputs, results = root / 'input', root / 'results'
    inputs.mkdir(); results.mkdir()
    status = dict(status='PREPARING', experiment=f'hunl_positions_p{checkpoint_hours}',
                  policies=specs, library_sha256=LIBRARY_SHA256, partition='development',
                  expected_positions_per_policy=DEVELOPMENT_POSITIONS, training=False,
                  external_benchmarks=False, assessment_unlocked=False)
    def stage(name, **fields):
        status.update(status=name, **fields)
        write_json(results / 'batch_status.json', status)
        print(json.dumps(status), flush=True)
    stage('PREPARING')
    try:
        library = frozen_library()
        library_path = results / 'library.json'
        write_json(library_path, library)
        storage('cp', results / 'batch_status.json', uri + '/batch_status.json')
        selected, markers = [], {}
        for spec in specs:
            path, marker = fetch_policy(spec, inputs)
            selected.append((spec['label'], str(path)))
            markers[spec['label']] = marker
        write_json(results / 'policies.json', dict(policies=specs, publication_markers=markers))
        stage('AUDITING')
        report = run(library_path, selected, results / 'audit', partition='development',
                     unlock_assessment=False, max_seconds=AUDIT_SECONDS, threads=1)
        if report['status'] == 'COMPLETE':
            if (report['expected_positions_per_policy'] != DEVELOPMENT_POSITIONS or
                    set(report['summary']['policies']) != {s['label'] for s in specs} or
                    any(p['completed_positions'] != DEVELOPMENT_POSITIONS
                        for p in report['summary']['policies'].values())):
                raise ValueError('Completed audit has incorrect policy/position coverage')
        stage(report['status'], elapsed_seconds=report['elapsed_seconds'])
    except BaseException as exc:
        stage('INTERRUPTED' if isinstance(exc, KeyboardInterrupt) else 'FAILED',
              error_type=type(exc).__name__, reason=str(exc))
        raise
    finally:
        publish(results, uri)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output-uri', required=True)
    parser.add_argument('--checkpoint-hours', type=int, choices=(56, 80), required=True)
    def terminate(signum, frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise KeyboardInterrupt('Batch audit terminated')
    signal.signal(signal.SIGTERM, terminate)
    if run_worker(**vars(parser.parse_args()))['status'] != 'COMPLETE':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
