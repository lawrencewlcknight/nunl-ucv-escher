"""Production schedules, policy consistency and recovery without extraction."""
from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from hunl_escher.config import baseline_config, smoke_config, make_solver, validate_config
from hunl_escher.optimization import replay_scaled_updates
from hunl_escher.checkpoint_schedule import CheckpointSchedule
from hunl_escher.training_state import read_training_state


@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)


@pytest.mark.parametrize("rows,expected", [
    (0, 0), (2047, 0), (2048, 250), (33889, 250), (50000, 250),
    (50001, 251), (250000, 1250), (500000, 2500), (900000, 2500),
])
def test_critic_fill_schedule(rows, expected):
    assert replay_scaled_updates(rows, 500000, 250, 2500, 2048) == expected


def test_calibration_fill_and_inference_workers():
    assert replay_scaled_updates(67726, 1000000, 100, 2000, 2048) == 136
    assert replay_scaled_updates(1000000, 1000000, 100, 2000, 2048) == 2000
    assert replay_scaled_updates(100, 256, 250, 0, 8) == 0


@pytest.mark.parametrize("key,value", [
    ("critic_min_train_steps", 0), ("critic_min_train_steps", True),
    ("critic_min_train_steps", 2501), ("calibration_min_train_steps", 2001),
    ("regret_policy_gradient_clip_norm", None), ("regret_policy_gradient_clip_norm", True),
    ("regret_policy_gradient_clip_norm", float("nan")),
    ("regret_policy_gradient_clip_norm", float("inf")),
    ("regret_policy_gradient_clip_norm", -1),
])
def test_invalid_production_controls(key, value):
    config = baseline_config()
    config[key] = value
    with pytest.raises(ValueError, match=key):
        validate_config(config)


@pytest.mark.parametrize("argmax", [True, False])
def test_critic_bootstrap_matches_collection_for_signed_regrets(argmax, monkeypatch):
    solver = make_solver(smoke_config())
    # Negative fallback, positive RM, tied fallback with an illegal first action,
    # and an empty terminal mask. Both players have different predictions.
    scores = torch.tensor([[-3., -1., -2., -4., -5., -6., -7.],
                           [8., 2., 1., -1., -2., -3., -4.],
                           [99., -2., -2., -5., -5., -5., -5.],
                           [1e-20, 2e-20, 3e-20, 0., 0., 0., 0.],
                           [3., 2., 1., -1., -2., -3., -4.]])
    masks = torch.tensor([[1., 1., 1., 0., 0., 0., 0.],
                          [0., 1., 1., 1., 0., 0., 0.],
                          [0., 1., 1., 0., 0., 0., 0.],
                          [1., 1., 1., 1., 1., 1., 1.],
                          [0., 0., 0., 0., 0., 0., 0.]])
    scores = scores[:, :solver.action_size]
    masks = masks[:, :solver.action_size]
    for p, trainer in enumerate(solver.regret_trainers):
        trainer.use_regret_matching_argmax = argmax
        values = scores if p == 0 else scores.flip(-1)
        monkeypatch.setattr(trainer, "predict", lambda *args, values=values: values)
    for player in (0, 1):
        actual = solver.q_value_trainer.members[0]._batched_predictive_strategies(
            torch.zeros(5, solver.infostate_size), masks,
            torch.full((5,), player), 1)
        values = scores if player == 0 else scores.flip(-1)
        for row in range(4):
            legal = torch.nonzero(masks[row]).flatten().tolist()
            expected = solver.regret_trainers[player].regret_matching(values[row].numpy(), legal)
            np.testing.assert_allclose(actual[row].numpy(), expected, atol=1e-7)
        assert torch.equal(actual[-1], torch.zeros(solver.action_size))


def test_regret_gradients_are_actually_clipped_without_clipping_labels(monkeypatch):
    solver = make_solver(smoke_config())
    trainer = solver.regret_trainers[0]
    for _ in range(8):
        trainer.add_data(np.zeros(trainer.input_size), np.full(trainer.output_size, 1e6), np.ones(trainer.output_size), 1)
    step = trainer.optimizer.step
    observed = []
    def checked_step(*args, **kwargs):
        norm = torch.linalg.vector_norm(torch.stack([
            p.grad.norm() for p in trainer.model.parameters() if p.grad is not None]))
        observed.append(float(norm))
        return step(*args, **kwargs)
    monkeypatch.setattr(trainer.optimizer, "step", checked_step)
    loss = trainer.train_model(1)
    assert loss > 1e10
    assert len(observed) == trainer.train_steps
    assert max(observed) <= 10.00001
    assert trainer.last_fit_stats["clipped_updates"] == trainer.train_steps
    assert trainer.last_fit_stats["max_gradient_norm"] > 10


def test_empty_regret_batch_preserves_network():
    trainer = make_solver(smoke_config()).regret_trainers[0]
    before = deepcopy(trainer.model.state_dict())
    assert trainer.train_model(1) is None
    assert not trainer.optimizer.state
    for key, value in before.items():
        assert torch.equal(value, trainer.model.state_dict()[key])


def test_underfilled_control_fits_do_not_advance_targets_or_optimizer():
    solver = make_solver(smoke_config())
    for trainer in [*solver.q_value_trainer.members, solver.calibration_trainer]:
        version = trainer.target_version
        before = deepcopy(trainer.target_model.state_dict())
        loss = (trainer.train_model(1) if hasattr(trainer, "target_history")
                else trainer.train_model())
        assert loss is None and trainer.last_fit_stats["updates"] == 0
        assert trainer.target_version == version and not trainer.optimizer.state
        if hasattr(trainer, "target_history"):
            assert not trainer.target_history
        for key, value in before.items():
            assert torch.equal(value, trainer.target_model.state_dict()[key])


@pytest.mark.parametrize("cache", [True, False])
def test_real_fits_follow_fill_schedule_and_keep_maximum(cache):
    config = smoke_config()
    config.update(baseline_network_train_steps=6, calibration_train_steps=8,
                  critic_min_train_steps=2, calibration_min_train_steps=2,
                  cache_frozen_critic_targets=cache)
    solver = make_solver(config, seed=71)
    learners = [*solver.q_value_trainer.members, solver.calibration_trainer]
    previous = [0, 0, 0]
    for _ in range(4):
        solver.iteration()
        for index, trainer in enumerate(learners):
            stats = trainer.last_fit_stats
            expected = replay_scaled_updates(len(trainer.buffer), trainer.buffer_size,
                trainer.minimum_train_steps, trainer.train_steps, trainer.batch_size)
            assert stats["updates"] == expected
            # Adam's actual step counts, not just scheduler-reported metadata.
            actual = {int(state["step"].item()) for state in trainer.optimizer.state.values()}
            if expected:
                assert actual == {previous[index] + expected}
            previous[index] += expected
            assert trainer.train_steps == (8 if index == 2 else 6)
    assert len(solver.q_value_trainer.members[0].target_history) == 4


def test_recovery_clock_and_policy_cadence_are_independent():
    schedule = CheckpointSchedule(last_recovery_wall_seconds=1000)
    options = dict(iteration=1, final=False, recovery_seconds=21600, policy_seconds=21600)
    assert schedule.due(active_seconds=100, wall_seconds=22599, **options) == (False, False)
    # Wall recovery is due despite only 100 seconds of active learning.
    assert schedule.due(active_seconds=100, wall_seconds=22600, **options) == (True, False)
    schedule.completed(100, wall_seconds=22600, recovery=True, policy=False)
    restored = CheckpointSchedule(**json.loads(json.dumps(schedule.state())))
    assert restored.last_policy_seconds == 0
    assert restored.due(active_seconds=21599, wall_seconds=22601, **options) == (False, False)
    assert restored.due(active_seconds=21600, wall_seconds=22601, **options) == (True, True)
    restored.completed(21600, wall_seconds=22601, recovery=True, policy=True)
    assert restored.due(active_seconds=21601, wall_seconds=22602, **options) == (False, False)
    assert restored.due(active_seconds=21601, wall_seconds=44200, **options) == (True, False)
    # Only timed recovery is suppressed by the recent-save rule.
    restored.completed(21601, wall_seconds=44200, recovery=True, policy=False)
    assert restored.due(active_seconds=21601, wall_seconds=44201,
                        **dict(options, final=True)) == (True, True)
    assert restored.due(active_seconds=21601, wall_seconds=44201,
                        checkpoint_every=1, **options) == (True, False)
    assert restored.due(active_seconds=21601, wall_seconds=44201,
                        policy_every=1, **options) == (True, True)


@pytest.mark.parametrize("age,expected", [(3599, False), (3600, True), (3601, True)])
def test_recovery_recent_save_boundary(age, expected):
    schedule = CheckpointSchedule(last_recovery_wall_seconds=0)
    schedule.completed(1, wall_seconds=21600-age, recovery=True, policy=False)
    assert schedule.due(iteration=1, final=False, active_seconds=1,
                        wall_seconds=21600, recovery_seconds=21600,
                        policy_seconds=21600) == (expected, False)
    assert schedule.last_recovery_check_wall_seconds == 21600


def test_skipped_slot_and_fixed_grid_survive_resume_and_downtime():
    options = dict(iteration=1, final=False, active_seconds=1,
                   recovery_seconds=21600, policy_seconds=21600)
    schedule = CheckpointSchedule(last_recovery_wall_seconds=0)
    schedule.completed(1, wall_seconds=19800, recovery=True, policy=True)
    assert schedule.due(wall_seconds=21600, **options) == (False, False)
    schedule = CheckpointSchedule(**json.loads(json.dumps(schedule.state())))
    # No retry at hour 7, even though the recent-save window expired.
    assert schedule.due(wall_seconds=25200, **options) == (False, False)
    assert schedule.due(wall_seconds=43200, **options) == (True, False)
    schedule.completed(1, wall_seconds=43320, recovery=True, policy=False)
    # Long downtime coalesces missed slots; the grid does not drift to hour 25.
    assert schedule.due(wall_seconds=90000, **options) == (True, False)
    assert schedule.last_recovery_check_wall_seconds == 86400
    schedule.completed(1, wall_seconds=90000, recovery=True, policy=False)
    assert schedule.due(wall_seconds=90001, **options) == (False, False)
    assert schedule.due(wall_seconds=108000, **options) == (True, False)


def test_disabled_timed_recovery_does_not_consume_slots():
    schedule = CheckpointSchedule(last_recovery_wall_seconds=0)
    assert schedule.due(iteration=1, final=False, active_seconds=1, wall_seconds=90000,
                        recovery_seconds=0, policy_seconds=21600) == (False, False)
    assert schedule.last_recovery_check_wall_seconds == 0


def assert_tree_equal(left, right):
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_tree_equal(a, b)
    elif isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    else:
        assert left == right


@pytest.mark.parametrize("policy_every", [0, 1])
def test_resume_after_recovery_matches_uninterrupted(tmp_path, monkeypatch, policy_every):
    import hunl_escher.train as runner
    original = runner.make_solver
    args = ["--smoke", "--iterations", "2", "--policy-every", str(policy_every),
            "--checkpoint-every", "1", "--checkpoint-seconds", "0", "--seed", "43"]
    control, interrupted = tmp_path / "control", tmp_path / "interrupted"
    runner.main([*args, "--output", str(control)])
    def crashing_solver(*a, **kw):
        solver = original(*a, **kw)
        iterate = solver.iteration
        def iteration():
            if solver.num_iteration == 1:
                raise RuntimeError("simulated interruption after recovery save")
            return iterate()
        solver.iteration = iteration
        return solver
    monkeypatch.setattr(runner, "make_solver", crashing_solver)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        runner.main([*args, "--output", str(interrupted)])
    assert json.loads((interrupted / "latest.json").read_text())["iteration"] == 1
    assert bool(list(interrupted.glob("policy_*.pkl"))) == bool(policy_every)
    monkeypatch.setattr(runner, "make_solver", original)
    runner.main([*args, "--output", str(interrupted), "--resume"])
    expected = read_training_state(control / "state_000002")
    actual = read_training_state(interrupted / "state_000002")
    for key in ("regret_trainers", "average_policy_trainer", "q_ensemble", "calibration",
                "rng", "replay_rng"):
        assert_tree_equal(expected[key], actual[key])
    first = json.loads((interrupted / "metrics_000001.json").read_text())
    assert first["policy_fitted"] == bool(policy_every) and first["recovery_saved"] is True


def test_skipped_check_is_persisted_without_rewriting_replay(tmp_path, monkeypatch):
    from hunl_escher import train
    now = [0.0]
    resuming = [False]
    monkeypatch.setattr(train.time, 'time', lambda: now[0])
    make = train.make_solver
    def make_with_clock(*args, **kwargs):
        solver = make(*args, **kwargs)
        iterate = solver.iteration
        def iteration():
            if solver.num_iteration == 0:
                now[0] = 19800  # Incidental recovery at 5.5 hours.
            elif solver.num_iteration == 1:
                now[0] = 25200 if resuming[0] else 21600
            elif not resuming[0]:
                raise RuntimeError('interruption after skipped check')
            else:
                now[0] = 43200
            return iterate()
        solver.iteration = iteration
        return solver
    monkeypatch.setattr(train, 'make_solver', make_with_clock)
    due = train.CheckpointSchedule.due
    def with_incidental_save(self, **kwargs):
        result = due(self, **kwargs)
        return (True, False) if kwargs['iteration'] == 1 else result
    monkeypatch.setattr(train.CheckpointSchedule, 'due', with_incidental_save)
    args = ['--smoke', '--iterations', '3', '--policy-every', '0', '--output', str(tmp_path)]
    with pytest.raises(RuntimeError, match='interruption after skipped check'):
        train.main(args)
    pointer = json.loads((tmp_path / 'latest.json').read_text())
    assert pointer['iteration'] == 1
    assert pointer['checkpoint_schedule']['last_recovery_check_wall_seconds'] == 21600
    assert pointer['checkpoint_schedule']['last_recovery_wall_seconds'] == 19800
    assert json.loads((tmp_path / 'manifest.json').read_text())['recovery_seconds'] == 21600
    assert sorted(p.name for p in tmp_path.glob('state_*')) == ['state_000001']
    # The full state was saved BEFORE the skipped check; only the pointer
    # has the updated grid. Resume must load the pointer's schedule.
    saved = read_training_state(tmp_path / 'state_000001')
    assert saved['checkpoint_schedule']['last_recovery_check_wall_seconds'] == 0
    resuming[0] = True
    now[0] = 25200
    train.main([*args, '--resume'])
    assert json.loads((tmp_path / 'metrics_000002.json').read_text())['recovery_saved'] is False
    assert sorted(p.name for p in tmp_path.glob('state_*')) == ['state_000003']


def test_run_start_is_not_treated_as_a_successful_checkpoint():
    schedule = CheckpointSchedule(last_recovery_check_wall_seconds=0)
    # A shorter explicit interval must not suppress the first ever save.
    assert schedule.due(iteration=1, final=False, active_seconds=1, wall_seconds=600,
                        recovery_seconds=600, policy_seconds=21600) == (True, False)
