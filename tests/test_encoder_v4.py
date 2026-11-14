"""Cross-encoder transfers, information retention and saved-policy compatibility."""
from itertools import combinations
import json
from pathlib import Path
import pickle
import random

import numpy as np
import pytest
import torch

from hunl_escher.features import HUNLFeatureEncoder, LegacyHUNLFeatureEncoder, StrategicV4HUNLFeatureEncoder, StructuredHUNLMLP
from hunl_escher import strategic_features as sf
from hunl_escher import legacy_strategic_features as old_sf
from hunl_escher.preflop_equity import preflop_equity, _equity, _estimate, _table
from hunl_escher.game import load_hunl_game
from test_hunl import state_at_street


def test_original_features_are_retained_with_separate_strategic_branch():
    old, new = LegacyHUNLFeatureEncoder(), HUNLFeatureEncoder()
    for street in range(4):
        state = state_at_street(street)
        for before, after, layout, prefix in (
            (old.information_state(state), new.information_state(state), new.policy_layout, new.policy_prefix_size),
            (old.full_state(state), new.full_state(state), new.full_state_layout, new.critic_prefix_size),
        ):
            split = layout.card_size + layout.strategic_size
            np.testing.assert_array_equal(before[:prefix], np.r_[after[:layout.card_size],
                after[split:split + prefix-layout.card_size]])
        view = state.decision_view()
        np.testing.assert_array_equal(sf.card_features(view.hole_cards, view.board_cards)[:49],
                                      old_sf.card_features(view.hole_cards, view.board_cards))


def test_next_categories_are_exact_and_distinct_from_completion():
    for board in ((0,20,29), (0,20,29,35)):
        hole = (48,44)
        f = dict(zip(sf.CARD_NAMES, sf.card_features(hole,board)))
        remaining = [c for c in range(52) if c not in hole + board]
        cats = sf.rank_cards([hole+board+(c,) for c in remaining])[1]
        for cat, name in enumerate(sf.CATEGORIES):
            assert f['next_card_'+name] == pytest.approx(np.mean(cats == cat))
        assert f['next_card_available'] == 1
    f = dict(zip(sf.CARD_NAMES,sf.card_features((48,44),(0,20,28))))
    assert f['next_flush_completion'] == 0  # Already a flush.
    assert f['next_card_flush'] > 0
    for board in ((),(0,20,28,33,39)):
        f = dict(zip(sf.CARD_NAMES,sf.card_features((48,44),board)))
        assert f['next_card_available'] == 0
        assert sum(f['next_card_'+c] for c in sf.CATEGORIES) == 0


def test_preflop_cache_is_169_classes_deterministic_and_rng_isolated():
    python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.random.get_rng_state()
    _equity.cache_clear()
    _table.cache_clear()
    for hole in combinations(range(52),2):
        equity, se = preflop_equity(hole)
        assert 0 <= equity <= 1 and 0 <= se <= .008
    assert _equity.cache_info().currsize == _equity.cache_info().misses == 169
    expected = preflop_equity((48,49))
    _equity.cache_clear()
    assert preflop_equity((48,49)) == expected
    assert random.getstate() == python_rng
    current = np.random.get_state()
    np.testing.assert_array_equal(numpy_rng[1], current[1])
    assert numpy_rng[0] == current[0] and numpy_rng[2:] == current[2:]
    assert torch.equal(torch_rng,torch.random.get_rng_state())
    # Independently regenerate representative packaged estimates offline.
    for hole in ((0,1),(0,48),(0,49),(44,48),(48,49)):
        np.testing.assert_array_equal(np.asarray(_estimate(hole),dtype=np.float32), _table()[hole])


def test_privileged_comparison_matches_native_showdown_and_policy_is_hidden_safe():
    encoder = HUNLFeatureEncoder()
    rng = np.random.default_rng(472)
    for _ in range(60):
        ids = tuple(map(int,rng.choice(52,9,replace=False)))
        state = state_at_street(3,holes=ids[:4],board=ids[4:])
        encoded = encoder.full_state(state)
        offset = 260 + 2*len(sf.CARD_NAMES)
        comparison = encoded[offset:offset+4]
        state.apply_action(1); state.apply_action(1)
        payoff = state.returns()[0]
        np.testing.assert_array_equal(comparison, (1,payoff>0,payoff==0,payoff<0))
    a = state_at_street(2)
    b = state_at_street(2,holes=(1,6,10,15))  # Only nonactor's hand differs.
    np.testing.assert_array_equal(encoder.information_state(a),encoder.information_state(b))


def test_aggression_handles_off_menu_bets_street_resets_and_observer():
    state = state_at_street()
    state.apply_native_action(347)
    state.apply_native_action(923)
    state.apply_action(1)
    for c in (20,25,30): state.apply_action(c)
    state.apply_action(1)  # BB checks flop; SB bets an off-menu amount.
    state.apply_native_action(1234)
    for observer in (0,1):
        view = state.decision_view(observer)
        f = dict(zip(sf.economics_names(view.action_menu),sf.betting_features(view,
                      sf.card_features(view.hole_cards,view.board_cards))))
        assert f['street_0_raise_count_share'] == pytest.approx(2/3)
        assert f['street_1_raise_count_share'] == .5
        assert f['street_2_raise_count_share'] == f['street_3_raise_count_share'] == 0
        assert f['street_0_aggressor_self'] == (observer == 1)
        assert f['street_1_aggressor_self'] == (observer == 0)
    counts, aggressors = sf.public_aggression(tuple(state.decision_view().public_history))
    assert counts == (2,1,0,0) and aggressors == (1,0,-1,-1)


@pytest.mark.parametrize("encoder_class", [LegacyHUNLFeatureEncoder, StrategicV4HUNLFeatureEncoder])
def test_v3_checkpoint_preserves_predictions_and_cannot_resume_as_v4(tmp_path, encoder_class):
    from hunl_escher.checkpointing import LoadedHUNLPolicy, CHECKPOINT_TYPE, CHECKPOINT_VERSION
    from hunl_escher.training_state import restore_training_state
    from hunl_escher.config import make_solver, smoke_config
    old = encoder_class()
    model = StructuredHUNLMLP(old.policy_layout,[16],7,branch_width=8)
    with torch.no_grad():
        for parameter in model.parameters(): parameter.uniform_(-.1,.1)
    game = load_hunl_game()
    path = tmp_path/'v3.pkl'
    path.write_bytes(pickle.dumps({'type':CHECKPOINT_TYPE,'version':CHECKPOINT_VERSION,
        'game':game.definition(),'feature_encoder':old.metadata(),
        'policy_model':model.checkpoint_metadata(),'policy_state_dict':model.state_dict()}))
    loaded = LoadedHUNLPolicy(game,path)
    for street in range(4):
        s = state_at_street(street)
        with torch.no_grad():
            logits = model(torch.tensor(old.information_state(s)))
            expected = torch.softmax(logits.masked_fill(~torch.tensor(s.legal_actions_mask(),dtype=torch.bool),-1e21),-1)
        for a,p in loaded.action_probabilities(s).items(): assert p == expected[a].item()
    solver = make_solver(smoke_config())
    with pytest.raises(ValueError,match='encoder mismatch'):
        restore_training_state(solver,{'game':game.definition(),'encoder':old.metadata()})


@pytest.mark.parametrize("fixture_name", ["holdem_card_contract_v1.json", "holdem_card_contract_v2.json"])
def test_shared_holdem_card_contract(fixture_name):
    fixture = json.loads((Path(__file__).parent/'fixtures'/fixture_name).read_text())
    for row in fixture['cases']:
        features = dict(zip(sf.CARD_NAMES,sf.card_features(tuple(row['hole']),tuple(row['board']))))
        for field,value in row['values'].items():
            assert features[fixture['mapping'][field]['no_limit']] == value
