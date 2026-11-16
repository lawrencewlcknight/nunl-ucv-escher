"""Download an existing P8 recovery, continue, then evaluate; never retrain P8."""
import argparse
import json
from pathlib import Path
import subprocess
import tarfile
import threading
from contextlib import contextmanager

from .checkpointing import sha256_file
from .cloud_worker import sync_output
from .continuation import checked_pointer, run_continuation
from .continuation_evaluate import run_suite
from .train import write_json
from .continuation_benchmark import run_final_benchmark
from .continuation_evaluate import verified_checkpoint


def storage(*args):
    subprocess.run(['gcloud','storage',*map(str,args)],check=True,timeout=1800)


def extract_source(archive, destination):
    destination=Path(destination)
    destination.mkdir(parents=True,exist_ok=False)
    with tarfile.open(archive,'r:gz') as handle:
        for member in handle.getmembers():
            path=Path(member.name)
            if path.is_absolute() or '..' in path.parts or not member.isfile():
                raise ValueError('Source archive must contain only safe regular files')
            target=destination/path
            target.parent.mkdir(parents=True,exist_ok=True)
            with handle.extractfile(member) as source, target.open('xb') as out:
                import shutil
                shutil.copyfileobj(source,out)


def download_source(uri, workspace, expected_p8_sha256):
    root=Path(workspace)
    root.mkdir(parents=True,exist_ok=False)
    source=root/'original-run'
    (source/'training').mkdir(parents=True)
    for name in ('P8.json','P8.pkl','training/manifest.json','training/latest.json','source.tar.gz'):
        storage('cp',uri.rstrip('/')+'/'+name,source/name)
    if sha256_file(source/'P8.pkl') != expected_p8_sha256:
        raise ValueError('Downloaded P8 does not match requested baseline')
    pointer=checked_pointer(source/'training')
    state=source/'training'/pointer['state']
    state.mkdir()
    storage('rsync','--recursive',uri.rstrip('/')+'/training/'+pointer['state'],state)
    trainer=root/'archived-trainer'
    extract_source(source/'source.tar.gz',trainer)
    return source,trainer


@contextmanager
def publisher(output, uri, *, training=False):
    """Periodic report uploads; checkpoint publication remains transactional."""
    stop = threading.Event()
    def sync(final=False):
        if not output.exists():
            return
        exclude = (r'^training/(?:state_[0-9]+(?:/.*)?|manifest\.json|latest\.json)$'
                   r'|^P[0-9]+\.(?:pkl|json)$|^cloud_checkpoint\.json$') if training else r'^$'
        if not final:
            exclude += r'|\.jsonl$'
        if sync_output(output, uri, exclude=exclude) != 0:
            raise RuntimeError('Continuation report upload failed')
    def loop():
        while not stop.wait(120):
            try:
                sync()
            except Exception:
                pass  # The mandatory final sync propagates errors.
    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    try:
        yield sync
    finally:
        stop.set()
        thread.join(timeout=1250)
        sync(final=True)


def run_cloud(workspace, source_uri, output_uri, expected_p8_sha256):
    source_uri, output_uri = source_uri.rstrip('/'), output_uri.rstrip('/')
    if source_uri == output_uri or output_uri.startswith(source_uri+'/') or source_uri.startswith(output_uri+'/'):
        raise ValueError('Continuation must use a separate sibling cloud prefix')
    root = Path(workspace)
    source, trainer = download_source(source_uri, root, expected_p8_sha256)
    contract = json.loads((source/'training/manifest.json').read_text())
    if contract['workers'] != 8 or contract['threads'] != 8:
        raise ValueError('This cloud plan requires the original eight-worker/eight-thread contract')
    storage('cp', '--if-generation-match=0', source/'source.tar.gz', output_uri+'/training_source.tar.gz')
    output = root/'run'
    with publisher(output, output_uri, training=True) as sync:
        try:
            report = run_continuation(source, trainer, output, upload_uri=output_uri)
            write_json(output/'provenance.json', dict(source_uri=source_uri, output_uri=output_uri,
                expected_p8_sha256=expected_p8_sha256,
                original_source_sha256=sha256_file(source/'source.tar.gz'),
                training_source_fingerprint=report['training_source_fingerprint']))
            write_json(output/'resume.json', dict(training_source_archive=output_uri+'/training_source.tar.gz',
                training_manifest=output_uri+'/training/manifest.json',
                latest_pointer=output_uri+'/training/latest.json',
                source_fingerprint=report['training_source_fingerprint'],
                instructions='Restore latest full state and manifest using the archived trainer and exact runtime; preserve all contract arguments for hunl_escher.train --resume. A playable policy alone cannot resume training.'))
            sync()
            evaluation = run_suite(output, output/'evaluation', workers=8, max_seconds=14400)
            sync(final=True)
            # This marker is last: the small evaluator can now be started without waiting.
            write_json(output/'slumbot_ready.json', dict(status='READY', policy='P32',
                sha256=report['checkpoints']['P32']['sha256'], source_uri=source_uri,
                local_evaluation_status=evaluation['status']))
        except Exception as exc:
            output.mkdir(parents=True, exist_ok=True)
            write_json(output/'cloud_failure.json', dict(error_type=type(exc).__name__, reason=str(exc)))
            raise


def run_external(workspace, output_uri, expected_p32_sha256):
    """Start only after publication; download policies/metadata, never full replay."""
    root = Path(workspace)
    root.mkdir(parents=True, exist_ok=False)
    inputs, output = root/'input', root/'slumbot'
    inputs.mkdir()
    uri = output_uri.rstrip('/')
    for name in ('slumbot_ready.json', 'continuation.json', 'P32.json', 'P32.pkl', 'P8.json'):
        storage('cp', uri+'/'+name, inputs/name)
    ready = json.loads((inputs/'slumbot_ready.json').read_text())
    policy, marker = verified_checkpoint(inputs, 'P32')
    if ready['status'] != 'READY' or ready['sha256'] != marker['sha256'] or sha256_file(policy) != expected_p32_sha256:
        raise ValueError('P32 identity differs from the approved ready marker')
    historical = inputs/'historical'
    historical.mkdir()
    report_path = None
    try:
        source_uri = ready['source_uri'].rstrip('/')
        for remote, local in [('benchmarks/slumbot/report.json', 'report.json'), ('P8.json', 'P8.json')]:
            storage('cp', source_uri+'/'+remote, historical/local)
        report_path = historical/'report.json'
    except (subprocess.SubprocessError, OSError):
        pass  # Optional reference must not block the new benchmark.
    with publisher(output, uri+'/benchmarks/slumbot'):
        report = run_final_benchmark(inputs, output, historical_report=report_path)
        if report_path:
            # Archive the exact report/identity used, excluding pilot data from inference.
            import shutil
            shutil.copytree(historical, output/'historical')
    if report['status'] != 'COMPLETE':
        raise RuntimeError('Slumbot evaluation incomplete; see published report and journal')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--role', choices=('train', 'slumbot'), default='train')
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output-uri', required=True)
    parser.add_argument('--source-uri')
    parser.add_argument('--expected-p8-sha256')
    parser.add_argument('--expected-p32-sha256')
    args = parser.parse_args()
    if args.role == 'train':
        if not args.source_uri or not args.expected_p8_sha256:
            parser.error('Training requires source URI and expected P8 hash')
        run_cloud(args.workspace, args.source_uri, args.output_uri, args.expected_p8_sha256)
    else:
        if not args.expected_p32_sha256:
            parser.error('Slumbot requires expected P32 hash')
        run_external(args.workspace, args.output_uri, args.expected_p32_sha256)


if __name__ == '__main__':
    main()

