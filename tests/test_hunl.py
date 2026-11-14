from copy import deepcopy
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from hunl_escher.config import baseline_config, smoke_config, make_solver, validate_config
from hunl_escher.features import HUNLFeatureEncoder
from hunl_escher.game import DEFAULT_ACTION_MENU, HUNL_GAME_PARAMETERS, load_hunl_game
from hunl_escher.checkpointing import save_policy_checkpoint, LoadedHUNLPolicy
from hunl_escher.training_state import build_training_state, restore_training_state, save_training_state, read_training_state
from unbiased_escher.estimator import control_variate_advantage, residual_adaptive_sampling_policy


@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)


def state_at_street(street=0, holes=(0, 5, 10, 15), board=(20, 25, 30, 35, 40)):
    state = load_hunl_game().new_initial_state()
    for card in holes:
        state.apply_action(card)
    index = 0
    for round_index in range(street):
        state.apply_action(1)
        state.apply_action(1)
        for _ in range(3 if round_index == 0 else 1):
            state.apply_action(board[index])
            index += 1
    return state


def model_states(solver):
    models = [t.model for t in solver.regret_trainers]
    models += [solver.calibration_trainer.model, solver.calibration_trainer.target_model,
               solver.ave_policy_trainer.model]
    models += [m.model for m in solver.q_value_trainer.members]
    models += [m.target_model for m in solver.q_value_trainer.members]
    return [{k: v.detach().clone() for k, v in m.state_dict().items()} for m in models]


def assert_models_equal(a, b, *, exact=True):
    for left, right in zip(a, b, strict=True):
        for key in left:
            torch.testing.assert_close(left[key], right[key], atol=0 if exact else 1e-6,
                                       rtol=0 if exact else 1e-5)


def test_game_and_street_contract():
    assert HUNL_GAME_PARAMETERS["betting"] == "nolimit"
    assert HUNL_GAME_PARAMETERS["stack"] == "20000 20000"
    assert load_hunl_game().num_distinct_actions() == len(DEFAULT_ACTION_MENU)
    encoder = HUNLFeatureEncoder()
    for street, count in enumerate((0, 3, 4, 5)):
        state = state_at_street(street)
        assert state.current_player() == (0 if street == 0 else 1)
        assert sum(state.information_state_tensor(0)[54:106]) == count
        for value, size in ((encoder.information_state(state), encoder.policy_size),
                            (encoder.full_state(state), encoder.full_state_size)):
            assert value.shape == (size,)
            assert np.isfinite(value).all()
            assert value.min() >= -1 and value.max() <= 1
    state.apply_action(1)
    state.apply_action(1)
    assert state.is_terminal() and sum(state.returns()) == 0
    assert not encoder.full_state(state).any()


def test_pot_raise_allin_and_terminal_payoffs():
    state = state_at_street()
    state.apply_action(4)
    assert "r300" in state.information_state_string(1)
    assert HUNLFeatureEncoder().information_state(state).max() <= 1
    state.apply_action(0)
    assert state.is_terminal() and state.returns() == [100.0, -100.0]
    state = state_at_street()
    state.apply_action(6)
    assert set(state.legal_actions()) == {0, 1}
    state.apply_action(1)
    while state.is_chance_node():
        state.apply_action(state.chance_outcomes()[0][0])
    assert state.is_terminal() and sum(state.returns()) == 0
    assert abs(state.returns()[0]) <= 20000


def test_hidden_cards_do_not_leak_into_policy_and_suits_are_canonical():
    encoder = HUNLFeatureEncoder()
    a = state_at_street(2)
    b = state_at_street(2, holes=(1, 6, 10, 15))
    # Player 1 is acting. Changing only player 0's cards cannot change its view.
    np.testing.assert_array_equal(encoder.information_state(a), encoder.information_state(b))
    assert not np.array_equal(encoder.full_state(a), encoder.full_state(b))
    perm = (2, 0, 3, 1)
    mapped = load_hunl_game().new_initial_state()
    for event in a.native_history():
        action = 4 * (event.action // 4) + perm[event.action % 4] if event.player == -1 else event.action
        mapped.apply_native_action(action)
    np.testing.assert_array_equal(encoder.information_state(a), encoder.information_state(mapped))
    np.testing.assert_array_equal(encoder.full_state(a), encoder.full_state(mapped))


def test_turn_and_river_order_is_retained():
    encoder = HUNLFeatureEncoder()
    a = state_at_street(3)
    b = state_at_street(3, board=(20, 25, 30, 40, 35))
    np.testing.assert_array_equal(a.information_state_tensor(), b.information_state_tensor())
    assert not np.array_equal(encoder.information_state(a), encoder.information_state(b))


def test_control_variate_mean_and_sampler_support_for_four_actions():
    values = np.array([-0.2, 0.1, 0.7, -0.6])
    policy = np.array([0.1, 0.2, 0.4, 0.3])
    proposal = residual_adaptive_sampling_policy([1, 2, 8, 4], np.ones(4), uniform_floor_mass=.2)
    assert np.all(proposal >= .05)
    average = np.zeros(4)
    for action in range(4):
        estimate = control_variate_advantage(np.array([2., -3., 1., .1]), beta=np.ones(4),
            sampled_action=action, sample_probability=proposal[action], sampled_return=values[action],
            policy=policy, legal_actions_mask=np.ones(4))
        average += proposal[action] * estimate.advantages
    np.testing.assert_allclose(average, values - policy @ values, atol=1e-14)


def test_role_specific_network_sizes():
    solver = make_solver(smoke_config())
    roles = [
        (solver.regret_trainers[0].model, 128, [256, 256, 256], 305159),
        (solver.regret_trainers[1].model, 128, [256, 256, 256], 305159),
        (solver.ave_policy_trainer.model, 128, [256, 256, 256], 305159),
        (solver.q_value_trainer.members[0].model, 96, [192, 192], 160231),
        (solver.q_value_trainer.members[1].model, 96, [192, 192], 160231),
        (solver.calibration_trainer.model, 64, [128, 128], 78722),
    ]
    for model, branch, layers, parameters in roles:
        metadata = model.checkpoint_metadata()
        assert metadata["branch_width"] == branch
        assert metadata["hidden_layers"] == layers
        assert sum(p.numel() for p in model.parameters()) == parameters
    assert sum(sum(p.numel() for p in m.parameters()) for m, *_ in roles) == 1314661


def custom_network_config():
    config = smoke_config()
    for role, branch, layers in (
        ("regret", 48, [96, 64, 32]), ("critic", 40, [80, 48]),
        ("calibration", 24, [48]), ("average_policy", 56, [112, 80]),
    ):
        config[f"{role}_branch_width"] = branch
        config[f"{role}_network_layers"] = layers
    return config


def test_role_sizes_are_independently_configurable():
    config = custom_network_config()
    solver = make_solver(config)
    roles = {
        "regret": [t.model for t in solver.regret_trainers],
        "critic": [model for m in solver.q_value_trainer.members
                   for model in (m.model, m.target_model)],
        "calibration": [solver.calibration_trainer.model, solver.calibration_trainer.target_model],
        "average_policy": [solver.ave_policy_trainer.model],
    }
    for role, models in roles.items():
        for model in models:
            metadata = model.checkpoint_metadata()
            assert metadata["branch_width"] == config[f"{role}_branch_width"]
            assert metadata["hidden_layers"] == config[f"{role}_network_layers"]
    solver.iteration()
    solver.train_average_policy()
    assert all(torch.isfinite(t).all() for state in model_states(solver) for t in state.values())


@pytest.mark.parametrize("role", ["regret", "critic", "calibration", "average_policy"])
@pytest.mark.parametrize("suffix,value", [
    ("branch_width", 0), ("branch_width", True), ("branch_width", 1.5),
    ("network_layers", []), ("network_layers", [64, -1]),
    ("network_layers", [True]), ("network_layers", [64.5]), ("network_layers", 128),
])
def test_invalid_architecture_rejected(role, suffix, value):
    config = smoke_config()
    key = f"{role}_{suffix}"
    config[key] = value
    with pytest.raises(ValueError, match=key):
        validate_config(config)


@pytest.mark.parametrize("old_policy_shape", [False, True])
def test_checkpoint_reloads_legal_policy(tmp_path, old_policy_shape):
    config = smoke_config()
    if old_policy_shape:
        # Saved model metadata, not today's defaults, controls inference shape.
        config["average_policy_branch_width"] = 64
        config["average_policy_network_layers"] = [192, 192]
    solver = make_solver(config)
    solver.iteration()
    solver.train_average_policy()
    path = save_policy_checkpoint(solver, tmp_path / "policy.pkl", seed=0,
                                  config=config, checkpoint_row={})
    loaded = LoadedHUNLPolicy(load_hunl_game(), path)
    for street in range(4):
        state = state_at_street(street)
        probs = loaded.action_probabilities(state)
        assert set(probs) == set(state.legal_actions())
        assert sum(probs.values()) == pytest.approx(1, abs=1e-6)
        encoded = torch.tensor(solver.feature_encoder.information_state(state))
        with torch.no_grad():
            logits = solver.ave_policy_trainer.model(encoded)
            mask = torch.tensor(state.legal_actions_mask(), dtype=torch.bool)
            expected = torch.softmax(logits.masked_fill(~mask, -1e21), -1).numpy()
        for action in probs:
            assert probs[action] == expected[action]


def test_cached_and_uncached_learner_agree():
    results = []
    for cache in (False, True):
        config = smoke_config()
        config["cache_frozen_critic_targets"] = cache
        solver = make_solver(config, seed=71)
        for _ in range(3):
            solver.iteration()
        results.append((solver.nodes_touched, model_states(solver)))
    assert results[0][0] == results[1][0]
    assert_models_equal(results[0][1], results[1][1], exact=False)


def test_complete_state_resume_matches_uninterrupted(tmp_path):
    config = smoke_config()
    solver = make_solver(config, seed=43)
    for _ in range(2):
        solver.iteration()
    kwargs = dict(seed=43, repository_commit="test", config=config)
    saved = build_training_state(solver, **kwargs, checkpoint_id="2", captured_checkpoints=[])
    save_training_state(tmp_path / "state", saved)
    solver.iteration()
    expected = model_states(solver)
    expected_nodes = solver.nodes_touched
    from hunl_escher.strategic_features import _card_features, _board_table, public_aggression
    from hunl_escher.preflop_equity import _equity
    for cache in (_card_features, _board_table, public_aggression, _equity):
        cache.cache_clear()
    restored = make_solver(config, seed=43)
    restore_training_state(restored, read_training_state(tmp_path / "state"), **kwargs)
    restored.iteration()
    assert restored.nodes_touched == expected_nodes
    assert_models_equal(expected, model_states(restored))
    payload = read_training_state(tmp_path / "state")
    with pytest.raises(ValueError, match="seed"):
        restore_training_state(restored, payload, **dict(kwargs, seed=44))
    changed = deepcopy(config)
    changed["critic_branch_width"] += 1
    with pytest.raises(ValueError, match="configuration"):
        restore_training_state(make_solver(changed, seed=43), payload,
                               **dict(kwargs, config=changed))


def test_repo_has_no_historical_experiments_and_baseline_is_reduced():
    root = Path(__file__).resolve().parents[1]
    assert not (root / "experiments").exists()
    assert not (root / "fhp_escher").exists()
    config = baseline_config()
    assert config["baseline_network_train_steps"] == 2500
    assert config["average_policy_train_steps"] == 20000
    assert config["cache_frozen_critic_targets"] is True


def test_cli_checkpoint_cadence_and_resume(tmp_path):
    from hunl_escher.train import main
    args = ["--smoke", "--checkpoint-every", "2", "--output", str(tmp_path)]
    main([*args, "--iterations", "2"])
    assert not (tmp_path / "policy_000001.pkl").exists()
    assert (tmp_path / "policy_000002.pkl").exists()
    first = json.loads((tmp_path / "metrics_000001.json").read_text())
    assert first["policy_fitted"] is False and first["policy_fit_seconds"] == 0
    main([*args, "--iterations", "3", "--resume"])
    last = json.loads((tmp_path / "metrics_000003.json").read_text())
    assert last["iteration"] == 3 and last["trajectories"] == 96
    assert last["training_elapsed_seconds"] > first["training_elapsed_seconds"] > 0
    assert last["iteration_wall_seconds"] >= last["learning_seconds"]
    with pytest.raises(ValueError, match="empty output"):
        main([*args, "--iterations", "4"])


@pytest.mark.ray
@pytest.mark.skipif(os.getenv("RUN_RAY_TESTS") != "1", reason="Set RUN_RAY_TESTS=1 for real actor tests")
@pytest.mark.parametrize("menu", [list(DEFAULT_ACTION_MENU), ["fold", "check_call", "pot", "all_in"]])
def test_real_parallel_collection_and_resume(tmp_path, menu):
    # Exercise both baseline sizes and independent, nonuniform role overrides.
    config = smoke_config() if menu == list(DEFAULT_ACTION_MENU) else custom_network_config()
    config["action_menu"] = menu
    kwargs = dict(seed=53, repository_commit="test", config=config)
    solver = make_solver(config, seed=53, workers=2)
    try:
        solver.iteration()
        assert solver.episode == 2 * config["num_traversals"]
        save_training_state(tmp_path / "state", build_training_state(
            solver, **kwargs, checkpoint_id="1", captured_checkpoints=[]))
        solver.iteration()
        expected, nodes = model_states(solver), solver.nodes_touched
    finally:
        solver.close()
    restored = make_solver(config, seed=53, workers=2)
    try:
        restore_training_state(restored, read_training_state(tmp_path / "state"), **kwargs)
        restored.iteration()
        assert restored.nodes_touched == nodes
        assert_models_equal(expected, model_states(restored))
    finally:
        restored.close()
