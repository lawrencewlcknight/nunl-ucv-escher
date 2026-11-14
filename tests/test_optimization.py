"""Production schedules, policy consistency and recovery without extraction."""
from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from hunl_escher.config import baseline_config, smoke_config, make_solver, validate_config
from hunl_escher.optimization import replay_scaled_updates
from hunl_escher.train import checkpoint_due
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
        assert torch.equal(actual[-1], torch.zeros(7))


def test_regret_gradients_are_actually_clipped_without_clipping_labels(monkeypatch):
    solver = make_solver(smoke_config())
    trainer = solver.regret_trainers[0]
    for _ in range(8):
        trainer.add_data(np.zeros(trainer.input_size), np.full(7, 1e6), np.ones(7), 1)
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
    options = dict(policy_every=50, checkpoint_every=0, checkpoint_seconds=1800)
    assert checkpoint_due(1, 100, seconds_since_save=1799, **options) == (False, False)
    assert checkpoint_due(1, 100, seconds_since_save=1800, **options) == (False, True)
    assert checkpoint_due(50, 100, seconds_since_save=10, **options) == (True, True)
    assert checkpoint_due(3, 3, seconds_since_save=10, **options) == (True, True)
    assert checkpoint_due(2, 100, policy_every=50, checkpoint_every=2,
                          checkpoint_seconds=0, seconds_since_save=0) == (False, True)


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


def test_resume_after_recovery_without_policy_matches_uninterrupted(tmp_path, monkeypatch):
    import hunl_escher.train as runner
    original = runner.make_solver
    args = ["--smoke", "--iterations", "2", "--policy-every", "50",
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
    assert not list(interrupted.glob("policy_*.pkl"))
    monkeypatch.setattr(runner, "make_solver", original)
    runner.main([*args, "--output", str(interrupted), "--resume"])
    expected = read_training_state(control / "state_000002")
    actual = read_training_state(interrupted / "state_000002")
    for key in ("regret_trainers", "average_policy_trainer", "q_ensemble", "calibration",
                "rng", "replay_rng"):
        assert_tree_equal(expected[key], actual[key])
    first = json.loads((interrupted / "metrics_000001.json").read_text())
    assert first["policy_fitted"] is False and first["recovery_saved"] is True
