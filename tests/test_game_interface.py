"""Interface regression tests against the independent native OpenSpiel state."""
from dataclasses import asdict

import numpy as np
import pyspiel
import pytest

from hunl_escher.config import make_solver, smoke_config
from hunl_escher.features import HUNLFeatureEncoder
from hunl_escher.game import HUNL_GAME_PARAMETERS, load_hunl_game
from hunl_escher.checkpointing import LoadedHUNLPolicy, save_policy_checkpoint
from unbiased_escher.estimator import control_variate_advantage, residual_adaptive_sampling_policy


def decision(holes=(0, 5, 10, 15), menu=None):
    game = load_hunl_game() if menu is None else load_hunl_game(menu)
    state = game.new_initial_state()
    for card in holes:
        state.apply_action(card)
    return state


def test_initial_snapshot_menu_and_duplicate_sizes():
    s = decision()
    v = s.decision_view()
    assert (v.player, v.actor, v.street) == (0, 0, 0)
    assert v.hole_cards == (0, 5) and v.board_cards == ()
    assert v.pot == 150 and v.to_call == 50
    assert v.stacks == (19950, 19900)
    assert v.contributions == v.street_contributions == (50, 100)
    assert (v.min_raise_to, v.max_raise_to) == (200, 20000)
    # min raise and half pot both produce 200; only the first is legal.
    assert v.native_actions == (0, 1, 200, None, 300, 500, 20000)
    assert s.native_action(4) == 300
    with pytest.raises(ValueError, match="Illegal learner"):
        s.apply_action(3)
    assert s.decision_view() == v


def test_off_menu_bets_are_exact_and_street_units_are_separate():
    s = decision()
    assert 275 not in s.decision_view().native_actions
    s.apply_street_raise_to(275)
    s.apply_action(1)
    for card in (20, 25, 30):
        s.apply_action(card)
    v = s.decision_view()
    assert v.pot == 550 and v.street_contributions == (0, 0)
    assert (v.player, v.street) == (1, 1)
    s.apply_street_raise_to(137)
    v = s.decision_view()
    assert v.contributions == (275, 412)
    assert v.street_contributions == (0, 137)
    assert v.pot == 687 and v.to_call == 137
    assert v.min_raise_to == 549
    assert s.native_history()[-1].action == 412
    # A pot raise is raise by the pot AFTER calling: 412 + (687 + 137).
    assert s.native_action(4) == 1236
    assert round(v.public_history[-1] * 20000) == 412
    alternate = s.clone()
    alternate.apply_native_action(550)
    assert s.decision_view() == v
    assert alternate.decision_view().contributions == (550, 412)


def test_short_allin_and_illegal_actions_leave_state_unchanged():
    s = decision()
    for invalid in (-1, 2, 199, 20001, 300.0, True):
        before = s.decision_view()
        with pytest.raises(ValueError):
            s.apply_native_action(invalid)
        assert s.decision_view() == before
    s.apply_native_action(10100)
    s.apply_native_action(20000)  # smaller than the previous 10,000 raise
    assert s.legal_actions() == [0, 1]
    assert s.decision_view().to_call == 9900
    s.apply_action(1)
    while s.is_chance_node():
        s.apply_action(s.chance_outcomes()[0][0])
    assert s.is_terminal() and sum(s.returns()) == 0
    assert s.legal_actions_mask() == [0] * 7
    with pytest.raises(ValueError):
        s.apply_action(1)


def test_hidden_cards_and_nonacting_traverser_view():
    encoder = HUNLFeatureEncoder()
    a, b = decision(), decision(holes=(0, 5, 18, 23))
    assert asdict(a.decision_view()) == asdict(b.decision_view())
    np.testing.assert_array_equal(encoder.information_state(a), encoder.information_state(b))
    assert not np.array_equal(encoder.full_state(a), encoder.full_state(b))
    a.apply_action(1)
    b.apply_action(1)
    assert a.current_player() == 1
    # Calibration may request traverser P0's view when P1 is acting.
    assert a.decision_view(0).hole_cards == (0, 5)
    assert a.decision_view(0).actor == 1
    np.testing.assert_array_equal(encoder.information_state(a, 0), encoder.information_state(b, 0))
    # Own cards are included, so P1's view can differ.
    assert not np.array_equal(encoder.information_state(a, 1), encoder.information_state(b, 1))


def test_random_exact_games_match_native_engine_and_preserve_history():
    rng = np.random.default_rng(289)
    game = load_hunl_game()
    native_game = pyspiel.load_game("universal_poker", dict(HUNL_GAME_PARAMETERS))
    encoder = HUNLFeatureEncoder()
    for _ in range(80):
        s, oracle = game.new_initial_state(), native_game.new_initial_state()
        while not oracle.is_terminal():
            assert s.current_player() == oracle.current_player()
            if oracle.is_chance_node():
                action = int(rng.choice(oracle.legal_actions()))
            else:
                v = s.decision_view()
                legal = oracle.legal_actions()
                assert all(a in legal for a in v.native_actions if a is not None)
                assert len(set(a for a in v.native_actions if a is not None)) == sum(v.legal_mask)
                assert v.contributions == tuple(oracle.observation_tensor(0)[106:])
                assert v.pot + sum(v.stacks) == 40000
                assert v.to_call == max(v.contributions) - v.contributions[v.actor]
                x = encoder.information_state(s)
                assert x.shape == (encoder.policy_size,) and np.isfinite(x).all()
                assert x.min() >= -1 and x.max() <= 1
                # Exercise calls/folds and arbitrary exact raises, not just menu sizes.
                action = int(rng.choice(legal if rng.random() < .5 else legal[:2]))
                history = oracle.full_history()
                for event, code in zip(history, v.public_history, strict=True):
                    recovered = 0 if code == -1 else 1 if code == -.5 else round(code * 20000)
                    if event.player == -1:
                        assert code == 0  # no card identity
                    else:
                        assert recovered == event.action
            s.apply_native_action(action)
            oracle.apply_action(action)
        assert s.is_terminal() and s.returns() == oracle.returns()


def test_long_min_raise_history_does_not_truncate():
    s = decision()
    encoder = HUNLFeatureEncoder()
    count = 0
    while not s.is_terminal():
        if s.is_chance_node():
            s.apply_action(s.chance_outcomes()[0][0])
            continue
        v = s.decision_view()
        assert len(v.public_history) <= encoder.history_slots
        assert encoder.information_state(s).shape == (encoder.policy_size,)
        s.apply_native_action(v.min_raise_to if v.min_raise_to is not None else 1)
        count += 1
    assert count >= 199
    while s.is_chance_node():
        s.apply_action(s.chance_outcomes()[0][0])
    assert s.is_terminal()
    assert len(s.native_history()) <= encoder.history_slots


def test_full_action_estimator_with_mask_and_seven_actions():
    mask = np.array(decision().legal_actions_mask())
    values = np.arange(7) / 7
    policy = mask / mask.sum()
    proposal = residual_adaptive_sampling_policy(np.arange(1, 8), mask, uniform_floor_mass=.2)
    assert np.all(proposal[mask == 0] == 0)
    assert np.all(proposal[mask == 1] >= .2 / mask.sum())
    mean = np.zeros(7)
    for action in np.flatnonzero(mask):
        mean += proposal[action] * control_variate_advantage(
            values + .3, beta=np.ones(7), sampled_action=action,
            sample_probability=proposal[action], sampled_return=values[action],
            policy=policy, legal_actions_mask=mask).advantages
    np.testing.assert_allclose(mean, (values - policy @ values) * mask, atol=1e-14)


def test_custom_menu_is_shared_by_solver_encoder_and_checkpoint(tmp_path):
    config = smoke_config()
    config["action_menu"] = ["fold", "check_call", "pot", "all_in"]
    solver = make_solver(config)
    assert solver.action_size == 4 and solver.feature_encoder.action_size == 4
    solver.iteration()
    solver.train_average_policy()
    path = save_policy_checkpoint(solver, tmp_path / "p.pkl", seed=0, config=config, checkpoint_row={})
    policy = LoadedHUNLPolicy(solver.game, path)
    probs = policy.action_probabilities(decision(menu=config["action_menu"]))
    assert sum(probs.values()) == pytest.approx(1, abs=1e-6)
    with pytest.raises(ValueError, match="menu"):
        LoadedHUNLPolicy(load_hunl_game(), path)
    with pytest.raises(ValueError, match="menu"):
        policy.action_probabilities(decision())
