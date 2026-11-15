"""Prove continuation across actual Python-process exits, using full networks."""
import json
from pathlib import Path
import subprocess
import sys

from hunl_escher.training_state import read_training_state
from test_optimization import assert_tree_equal


def test_fresh_process_resume_matches_uninterrupted_training(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    common = [sys.executable, '-m', 'hunl_escher.train', '--smoke',
              '--seed', '43', '--threads', '1', '--policy-every', '1',
              '--checkpoint-every', '1', '--recovery-seconds', '0']
    def run(output, iterations, resume=False):
        result = subprocess.run(
            [*common, '--output', str(output), '--iterations', str(iterations),
             *(['--resume'] if resume else [])],
            cwd=repository, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout[-6000:] + result.stderr[-6000:]

    control, continued = tmp_path / 'control', tmp_path / 'continued'
    run(control, 3)
    run(continued, 2)
    before = read_training_state(continued / 'state_000002')
    assert before['solver']['num_iteration'] == 2
    before_steps = [member['target_version'] for member in before['q_ensemble']['members']]
    # Release memory-mapped files before the resumed run prunes the old save.
    del before
    run(continued, 3, resume=True)
    pointer = json.loads((continued / 'latest.json').read_text())
    assert pointer['iteration'] == 3 and pointer['state'] == 'state_000003'
    assert sorted(path.name for path in continued.glob('state_*')) == ['state_000003']
    expected = read_training_state(control / 'state_000003')
    actual = read_training_state(continued / 'state_000003')
    for key in ('regret_trainers', 'average_policy_trainer', 'q_ensemble',
                'calibration', 'gate_controller', 'rng', 'replay_rng', 'actors'):
        assert_tree_equal(expected[key], actual[key])
    for key in ('num_iteration', 'episode', 'nodes_touched'):
        assert expected['solver'][key] == actual['solver'][key]
    for previous, member in zip(before_steps, actual['q_ensemble']['members']):
        assert member['target_version'] > previous
    assert actual['solver']['training_elapsed_seconds'] > 0
    # Both the checkpoint and latest pointer retain scheduler state. Wall and
    # measured learner durations intentionally differ between separate runs.
    assert actual['checkpoint_schedule']['last_recovery_check_wall_seconds'] is not None
    assert pointer['checkpoint_schedule']['last_policy_seconds'] > 0
