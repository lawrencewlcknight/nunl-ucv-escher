"""Execution optimisations must preserve the previous training trajectory."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from adaptive_escher import frozen_target_cache as targets
from hunl_escher import features
from hunl_escher.config import make_solver, smoke_config
from hunl_escher.training_state import build_training_state
from unbiased_escher import holdem_fitting
from vr_deep_cfr.solver import RegretTrainer
from test_hunl import state_at_street


def assert_tree_equal(left, right):
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_tree_equal(a, b)
    elif isinstance(left, (torch.Tensor, np.ndarray)):
        if isinstance(left, torch.Tensor):
            left, right = left.detach().cpu().numpy(), right.detach().cpu().numpy()
        assert left.shape == right.shape and left.dtype == right.dtype
        assert left.tobytes() == right.tobytes()
    else:
        np.testing.assert_equal(left, right)  # Includes matching NaN diagnostics.


def reference_suits(*channels):
    signatures = [tuple(float(x) for channel in channels for x in channel[suit])
                  for suit in range(4)]
    order = sorted(range(4), key=lambda suit: signatures[suit], reverse=True)
    return tuple(channel[order].reshape(-1) for channel in channels)


def reference_forward(self, model, x, mask):
    with torch.device(self.device):
        x = torch.as_tensor(x, dtype=torch.float32)
        mask = torch.as_tensor(mask, dtype=torch.float32)
        with torch.no_grad():
            return self.predict(model, x, mask).cpu().numpy()


def reference_batch(buffer, indices, device):
    floats = ('history', 'next_history', 'next_state', 'reward')
    integers = ('next_legal_actions_mask', 'next_player', 'action', 'done')
    return tuple(torch.as_tensor(getattr(buffer, name + '_buf')[indices],
                                 dtype=dtype, device=device)
                 for names, dtype in ((floats, torch.float32), (integers, torch.int64))
                 for name in names)


def install_reference(monkeypatch):
    """The pre-optimisation operations; no alternate learning configuration."""
    monkeypatch.setattr(features, '_canonicalise_suits', reference_suits)
    monkeypatch.setattr(features.LegacyHUNLFeatureEncoder, '_cached_state_features',
                        lambda self, state, key, build: build())
    monkeypatch.setattr(RegretTrainer, 'forward', reference_forward)
    monkeypatch.setattr(targets, 'transition_batch', reference_batch)
    for module in (targets, holdem_fitting):
        monkeypatch.setattr(module, 'target_batch_indices',
                            lambda start, stop, batch: np.resize(np.arange(start, stop), batch))


@pytest.mark.parametrize('channels_count', [4, 5])
def test_suit_order_matches_float_tuples_including_ties(channels_count):
    rng = np.random.default_rng(27)
    for _ in range(1000):
        channels = [rng.integers(0, 2, (4, 13)).astype(np.float32)
                    for _ in range(channels_count)]
        # Exercise stable ties and signed zero as well as dense binary patterns.
        for channel in channels:
            channel[1] = channel[0]
            channel[channel == 0] = -0.0
        actual = features._canonicalise_suits(*channels)
        expected = reference_suits(*channels)
        assert all(a.tobytes() == b.tobytes() for a, b in zip(actual, expected))


@pytest.mark.parametrize('encoder_type', [features.LegacyHUNLFeatureEncoder,
    features.StrategicV4HUNLFeatureEncoder, features.HUNLFeatureEncoder])
def test_cached_features_are_observer_specific_owned_and_invalidated(encoder_type):
    encoder = encoder_type()
    for street in range(4):
        state = state_at_street(street)
        for player in (0, 1):
            expected = encoder.encode_decision(state.decision_view(player))
            actual = encoder.information_state(state, player)
            np.testing.assert_array_equal(actual, expected)
            actual[:] = -999  # A caller must not corrupt cached features.
            np.testing.assert_array_equal(encoder.information_state(state, player), expected)
        full = encoder._encode_full_state(state)
        encoder.full_state(state)[:] = -999
        np.testing.assert_array_equal(encoder.full_state(state), full)
        before = {key: value.copy() for key, value in state._feature_cache.items()}
        clone = state.clone()
        assert not clone._feature_cache
        clone.apply_action(1)
        assert_tree_equal(state._feature_cache, before)
        # Use the native action entry point used by external opponents as well.
        state.apply_native_action(1)
        assert not state._feature_cache
        if not state.is_chance_node() and not state.is_terminal():
            for player in (0, 1):
                np.testing.assert_array_equal(encoder.information_state(state, player),
                    encoder.encode_decision(state.decision_view(player)))
        state = state_at_street(street)
        if 0 not in state.legal_actions():
            state.apply_native_action(state.decision_view().min_raise_to)
        encoder.full_state(state)
        state.apply_action(0)
        assert state.is_terminal() and not state._feature_cache
        np.testing.assert_array_equal(encoder.full_state(state), np.zeros(encoder.full_state_size))


def test_encoder_instances_and_native_off_menu_actions_do_not_reuse_stale_features():
    state = state_at_street()
    encoders = [features.LegacyHUNLFeatureEncoder(), features.StrategicV4HUNLFeatureEncoder(),
                features.HUNLFeatureEncoder(), features.HUNLFeatureEncoder()]
    for encoder in encoders:
        np.testing.assert_array_equal(encoder.information_state(state),
                                      encoder.encode_decision(state.decision_view()))
    assert len(state._feature_cache) == len(encoders)
    encoder = encoders[-1]
    old = encoder.information_state(state)
    assert 333 not in state.decision_view().native_actions
    state.apply_street_raise_to(333)
    assert not state._feature_cache
    actual = encoder.information_state(state)
    assert not np.array_equal(old, actual)
    np.testing.assert_array_equal(actual, encoder.encode_decision(state.decision_view()))
    # Calls close preflop, then chance cards must invalidate any state cache too.
    state.apply_action(1)
    assert state.is_chance_node()
    state._feature_cache['sentinel'] = old
    state.apply_action(state.chance_outcomes()[0][0])
    assert not state._feature_cache


def test_target_batches_share_full_rows_and_preserve_tail_padding():
    class Unused:
        def __getitem__(self, index):
            raise AssertionError('Unused current-state fields must not be gathered')
    data = np.arange(11 * 7, dtype=np.float32).reshape(11, 7)
    buffer = SimpleNamespace(history_buf=Unused(), action_buf=Unused(),
        next_history_buf=data, next_state_buf=data, reward_buf=data[:, 0],
        next_legal_actions_mask_buf=np.ones((11, 7), dtype=np.int64),
        next_player_buf=np.zeros(11, dtype=np.int64), done_buf=np.zeros(11, dtype=np.int64))
    for start, stop in ((0, 4), (4, 8), (8, 11)):
        indices = targets.target_batch_indices(start, stop, 4)
        batch = targets.transition_batch(buffer, indices, 'cpu')
        assert batch[0] is None and batch[6] is None
        expected = data[np.resize(np.arange(start, stop), 4)]
        np.testing.assert_array_equal(batch[1].numpy(), expected)
        assert np.shares_memory(batch[1].numpy(), data) == (stop - start == 4)


def training_snapshot(config):
    solver = make_solver(config, seed=771)
    snapshots = []
    for _ in range(3):
        solver.iteration()
        solver.train_average_policy()
        payload = build_training_state(solver, seed=771, checkpoint_id='equivalence',
            repository_commit='test', config=config, captured_checkpoints=[])
        for key in ('training_elapsed_seconds', 'checkpoint_rows'):
            payload['solver'].pop(key)
        # Recovery omits fresh-regret replay; compare it explicitly as well.
        payload['regret_replay'] = [{key: value[:len(t.buffer)].copy()
            for key, value in vars(t.buffer).items() if isinstance(value, np.ndarray)}
            for t in solver.regret_trainers]
        snapshots.append(deepcopy(payload))
    return snapshots


@pytest.mark.parametrize('threads', [1, 4])
def test_training_matches_reference_including_replay_rng_and_adam(monkeypatch, threads):
    torch.set_num_threads(threads)
    try:
        config = smoke_config()
        # Nondivisible capacities exercise padded tails and ring/reservoir
        # replacement; repeated fits exercise learned (nonzero) network weights.
        for key in ('advantage_buffer_size', 'ave_policy_buffer_size',
                    'baseline_buffer_size', 'calibration_buffer_size'):
            config[key] = 37
        config['num_traversals'] = 64
        actual = training_snapshot(config)
        with monkeypatch.context() as patch:
            install_reference(patch)
            expected = training_snapshot(config)
        assert_tree_equal(actual, expected)
    finally:
        torch.set_num_threads(1)


def test_full_size_minibatches_and_padded_labels_match_reference(monkeypatch):
    """Exercise the production GEMM shape, not just small smoke minibatches."""
    torch.set_num_threads(4)
    try:
        config = smoke_config()
        for key in ('advantage_buffer_size', 'ave_policy_buffer_size', 'calibration_buffer_size'):
            config[key] = 2051
        config['baseline_buffer_size'] = 4102  # 2051 per critic, with a 3-row tail.
        for key in ('advantage_batch_size', 'ave_policy_batch_size',
                    'baseline_batch_size', 'calibration_batch_size'):
            config[key] = 2048

        def fit():
            solver = make_solver(config, seed=731)
            rng = np.random.default_rng(998)
            for trainer in solver.regret_trainers:
                b = trainer.buffer
                b.cur_id = b.buffer_size
                b.infostate_buf[:] = rng.random(b.infostate_buf.shape, dtype=np.float32)
                b.q_value_buf[:] = rng.normal(size=b.q_value_buf.shape)
                b.q_value_mask_buf[:] = 1
                b.iteration_buf[:] = 1
            for member in solver.q_value_trainer.members:
                b = member.buffer
                b.size = b.buffer_size
                for field in ('history_buf', 'next_history_buf', 'next_state_buf', 'reward_buf'):
                    array = getattr(b, field)
                    array[:] = rng.random(array.shape, dtype=np.float32)
                b.action_buf[:] = np.arange(b.size) % solver.action_size
                b.next_legal_actions_mask_buf[:] = 1
                b.next_player_buf[:] = np.arange(b.size) % 2
                b.done_buf[:] = np.arange(b.size) % 3 == 0
            results = []
            for iteration in (1, 2):
                for trainer in solver.regret_trainers:
                    trainer.train_model(iteration)
                for member in solver.q_value_trainer.members:
                    results.append(targets.build_target_cache(member, iteration).clone())
                    member.train_model(iteration)
                results.extend(deepcopy((trainer.model.state_dict(), trainer.optimizer.state_dict()))
                    for trainer in [*solver.regret_trainers, *solver.q_value_trainer.members])
            results.append(solver._capture_rng_state())
            results.extend(deepcopy(trainer.buffer.rng.bit_generator.state)
                for trainer in [*solver.regret_trainers, *solver.q_value_trainer.members])
            return results

        actual = fit()
        with monkeypatch.context() as patch:
            install_reference(patch)
            expected = fit()
        assert_tree_equal(actual, expected)
    finally:
        torch.set_num_threads(1)
