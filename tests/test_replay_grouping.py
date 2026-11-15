"""Grouping preserves the weighted objective and never copies all features."""
from copy import deepcopy
import random

import numpy as np
import pytest
import torch

from unbiased_escher.efficient_replay import CompactReservoirBuffer
from unbiased_escher.holdem_fitting import policy_validation_mask, weighted_policy_ce
from unbiased_escher import replay_grouping as grouping


def replay(rows=103, capacity=103):
    rng = np.random.default_rng(4)
    buffer = CompactReservoirBuffer(capacity, 19, 3, seed=8)
    features = rng.integers(-3, 4, size=(31, 19)).astype(np.float32)
    # Encode two players and four streets at known columns for coverage tests.
    features[:, :2] = np.eye(2)[np.arange(31) % 2]
    features[:, 2:6] = np.eye(4)[np.arange(31) % 4]
    for i in range(rows):
        values = rng.random(3).astype(np.float32)
        values /= values.sum()
        buffer.add(features[i % 31], values, np.ones(3), 1 + i % 10)
    return buffer


def reference(buffer, iteration):
    n = min(buffer.cur_id, buffer.buffer_size)
    x, first, inverse = np.unique(buffer.infostate_buf[:n], axis=0,
                                  return_index=True, return_inverse=True)
    w = (np.maximum(buffer.iteration_buf[:n].astype(np.float64).ravel(), 0) / iteration * 2) ** 2
    mass = np.zeros(len(x))
    numerator = np.zeros((len(x), 3))
    np.add.at(mass, inverse, w)
    np.add.at(numerator, inverse, buffer.q_value_buf[:n].astype(np.float64) * w[:, None])
    return x, (numerator / mass[:, None]).astype(np.float32), buffer.q_value_mask_buf[first], (mass * len(x) / n).astype(np.float32)


@pytest.mark.parametrize("chunk", [1, 7, 4096])
@pytest.mark.parametrize("rows,capacity", [(103,103), (241,103)])
def test_matches_dense_reference_even_after_reservoir_overflow(chunk, rows, capacity):
    buffer = replay(rows, capacity)
    state = deepcopy(buffer.rng.bit_generator.state)
    numpy_rng = np.random.get_state()
    python_rng = random.getstate()
    torch_rng = torch.get_rng_state().clone()
    data, stats = grouping.grouped_policy_data(buffer, 10, 2, "cpu",
        context_columns=(slice(0,2), slice(2,6)), chunk_rows=chunk)
    features, targets, masks, weights = data
    expected = reference(buffer, 10)
    lookup = {row.tobytes(): i for i, row in enumerate(expected[0])}
    permutation = [lookup[row.tobytes()] for row in features.full_batch().numpy()]
    for actual, ref in zip(data[1:], expected[1:]):
        np.testing.assert_array_equal(actual.numpy(), ref[permutation])
    assert np.shares_memory(features.source, buffer.infostate_buf)
    assert stats["full_feature_copy_bytes"] == 0
    assert stats["seen"] == rows and stats["retained"] == min(rows,capacity)
    assert np.sum(stats["retained_by_player_street"]) == min(rows,capacity)
    assert np.sum(stats["unique_by_player_street"]) == len(features)
    assert grouping.retained_context_counts(buffer, (slice(0,2),slice(2,6))) == stats["retained_by_player_street"]
    assert state == buffer.rng.bit_generator.state
    assert python_rng == random.getstate()
    assert torch.equal(torch_rng, torch.get_rng_state())
    after = np.random.get_state()
    assert numpy_rng[0] == after[0] and numpy_rng[2:] == after[2:]
    np.testing.assert_array_equal(numpy_rng[1], after[1])


def test_hash_collisions_and_signed_zero_do_not_merge_distinct_information_sets(monkeypatch):
    buffer = replay()
    buffer.infostate_buf[:4] = 0
    buffer.infostate_buf[1, 8] = -0.0
    monkeypatch.setattr(grouping, "_fingerprints", lambda x, chunk: np.zeros(len(x),dtype="V16"))
    data, _ = grouping.grouped_policy_data(buffer, 10, 2, "cpu", chunk_rows=7)
    expected = reference(buffer, 10)
    np.testing.assert_array_equal(data[0].full_batch().numpy(), expected[0])
    for actual, ref in zip(data[1:], expected[1:]):
        np.testing.assert_array_equal(actual.numpy(), ref)


def test_masks_must_agree_and_invalid_rows_fail():
    buffer = replay()
    buffer.q_value_mask_buf[31, 0] = 0  # Same features as row zero.
    with pytest.raises(ValueError, match="masks differ"):
        grouping.grouped_policy_data(buffer, 10, 2, "cpu", chunk_rows=3)
    buffer.q_value_mask_buf[31, 0] = 1
    buffer.infostate_buf[2, 7] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        grouping.grouped_policy_data(buffer, 10, 2, "cpu")


def test_lazy_validation_loss_and_minibatch_gradients_match_dense():
    buffer = replay()
    (features, targets, masks, weights), _ = grouping.grouped_policy_data(buffer, 10, 2, "cpu")
    dense = features.full_batch()
    np.testing.assert_array_equal(policy_validation_mask(features,.2), policy_validation_mask(dense,.2))
    model = torch.nn.Linear(19, 3)
    ids = np.arange(len(features))
    assert weighted_policy_ce(model, features, targets, masks, weights, ids, 7) == pytest.approx(
        weighted_policy_ce(model, dense, targets, masks, weights, ids, 7))
    selected = torch.tensor([0,3,6,8])
    gradients=[]
    for inputs in (features[selected], dense[selected]):
        model.zero_grad()
        loss=(-(targets[selected]*torch.log_softmax(model(inputs),-1)).sum(-1)*weights[selected]).mean()
        loss.backward()
        gradients.append([p.grad.clone() for p in model.parameters()])
    for a,b in zip(*gradients):
        torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_coverage_weights_and_effective_sample_size():
    buffer = CompactReservoirBuffer(4, 6, 3)
    for iteration in (1,2,3,4):
        buffer.add([0,1,0,0,0,1], [1,0,0], [1,1,1], iteration)
    _, stats = grouping.grouped_policy_data(buffer, 4, 2, "cpu",
        context_columns=(slice(0,2),slice(2,6)))
    w=(np.arange(1,5)/4*2)**2
    assert stats['retained_by_player_street'][1][3] == 4
    assert stats['unique_by_player_street'][1][3] == 1
    assert stats['weight_mass_by_player_street'][1][3] == pytest.approx(w.sum())
    assert stats['effective_rows_by_player_street'][1][3] == pytest.approx(w.sum()**2/(w*w).sum())


def test_grouping_is_rebuilt_when_the_reservoir_changes():
    buffer = replay()
    old, _ = grouping.grouped_policy_data(buffer, 10, 2, "cpu")
    buffer.infostate_buf[0] = 12
    new, stats = grouping.grouped_policy_data(buffer, 10, 2, "cpu")
    assert len(new[0]) == len(old[0]) + 1
    assert stats['fit_iteration'] == 10


def test_empty_reservoir_counts_are_finite_and_grouping_rejects_it():
    buffer = CompactReservoirBuffer(8,19,3)
    assert grouping.reservoir_counts(buffer)['retained_fraction'] is None
    with pytest.raises(ValueError, match="empty"):
        grouping.grouped_policy_data(buffer,1,2,'cpu')


def test_legacy_float64_parent_buffer_keeps_float32_grouping_semantics():
    from vr_deep_cfr.solver import ReservoirBuffer
    buffer = ReservoirBuffer(4, 6, 3)
    buffer.add(np.ones(6), [1,0,0], [1,1,1], 1)
    buffer.add(np.ones(6) + 1e-10, [0,1,0], [1,1,1], 2)
    data, stats = grouping.grouped_policy_data(buffer, 2, 2, 'cpu')
    assert len(data[0]) == 1
    assert data[0][:].dtype == torch.float32
    np.testing.assert_allclose(data[1], [[.2,.8,0]])
    assert stats['full_feature_copy_bytes'] == 2 * 6 * 4
