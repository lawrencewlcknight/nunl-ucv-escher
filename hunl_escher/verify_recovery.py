"""Standalone fresh-process restore check, importing ONLY the archived trainer."""
import argparse
import json
from pathlib import Path
import platform
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--training', type=Path, required=True)
    p.add_argument('--policy', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    args = p.parse_args()
    sys.path.insert(0, str(args.source.resolve()))
    from importlib.metadata import version
    import numpy as np
    import torch
    from hunl_escher.train import source_fingerprint, write_json
    from hunl_escher.config import make_solver
    from hunl_escher.checkpointing import load_checkpoint_payload
    from hunl_escher.training_state import read_training_state, restore_training_state
    contract = json.loads((args.training / 'manifest.json').read_text())
    runtime = dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                   open_spiel=version('open_spiel'), ray=version('ray') if contract['workers'] > 1 else None)
    if runtime != contract['runtime'] or source_fingerprint() != contract['source_fingerprint']:
        raise ValueError('Archived training source/runtime differs from the saved contract')
    pointer = json.loads((args.training / 'latest.json').read_text())
    import re
    if not re.fullmatch(r'state_[0-9]{6,}', pointer['state']):
        raise ValueError('Invalid state path')
    payload = read_training_state(args.training / pointer['state'])
    torch.set_num_threads(contract['threads'])
    solver = make_solver(contract['config'], seed=contract['seed'], workers=contract['workers'])
    try:
        restore_training_state(solver, payload, seed=contract['seed'],
            repository_commit=contract['source_fingerprint'], config=contract['config'])
        policy = load_checkpoint_payload(args.policy)
        if solver.num_iteration != pointer['iteration'] or policy['outer_iteration'] != solver.num_iteration:
            raise ValueError('Recovery and policy iteration differ')
        if solver._training_elapsed_seconds() != policy['training_elapsed_seconds']:
            raise ValueError('Recovery and policy active time differ')
        actual = solver.ave_policy_trainer.model.state_dict()
        if actual.keys() != policy['policy_state_dict'].keys() or any(
            not torch.equal(actual[k].cpu(), v) or not torch.isfinite(v).all()
            for k,v in policy['policy_state_dict'].items()):
            raise ValueError('Restored average policy differs from playable export')
        write_json(args.report, dict(status='VERIFIED', iteration=solver.num_iteration,
            active_seconds=solver._training_elapsed_seconds(), workers=contract['workers'],
            source_fingerprint=contract['source_fingerprint'],
            verification='All shards checksum-verified; full solver, optimizers, replay and actor RNGs restored; playable policy matches exactly'))
    finally:
        if hasattr(solver, 'close'):
            solver.close()


if __name__ == '__main__':
    main()
