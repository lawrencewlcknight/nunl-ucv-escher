"""Independent best-five oracle, poker edge cases and information boundaries."""
from collections import Counter
from itertools import combinations, permutations

import numpy as np
import pytest

from hunl_escher.features import HUNLFeatureEncoder, encoder_from_metadata
from hunl_escher.strategic_features import (
    CARD_NAMES, card_features, rank_cards, betting_features, economics_names,
    _card_features, _board_table, HAND_CACHE_SIZE, BOARD_CACHE_SIZE,
)
from test_hunl import state_at_street


def cards(text):
    return tuple("23456789TJQKA".index(c[0]) * 4 + "cdhs".index(c[1]) for c in text.split())


def features(hole, board):
    return dict(zip(CARD_NAMES, card_features(cards(hole), cards(board)), strict=True))


def five_oracle(hand):
    """Scalar five-card rules, deliberately no shared lookup/evaluator code."""
    counts = Counter(c // 4 + 2 for c in hand)
    ranks = sorted(counts, reverse=True)
    flush = len({c % 4 for c in hand}) == 1
    straight = 5 if ranks == [14, 5, 4, 3, 2] else (
        ranks[0] if len(ranks) == 5 and ranks[0] - ranks[-1] == 4 else 0)
    groups = sorted(((n, r) for r, n in counts.items()), reverse=True)
    if flush and straight:
        result = (8, straight)
    elif groups[0][0] == 4:
        result = (7, groups[0][1], groups[1][1])
    elif [n for n, _ in groups] == [3, 2]:
        result = (6, groups[0][1], groups[1][1])
    elif flush:
        result = (5, *ranks)
    elif straight:
        result = (4, straight)
    else:
        category = {1: 0, 2: 2 if len(groups) == 3 else 1, 3: 3}[groups[0][0]]
        result = (category, *(r for _, r in groups))
    return (*result, *((0,) * (6 - len(result))))


def best_oracle(hand):
    return max(five_oracle(combo) for combo in combinations(hand, 5))


def test_vector_evaluator_matches_independent_best_five_oracle():
    rng = np.random.default_rng(20261114)
    for size in (5, 6, 7):
        hands = np.asarray([rng.choice(52, size, replace=False) for _ in range(1500)])
        ranks, cats, ties = rank_cards(hands)
        expected = np.asarray([best_oracle(hand) for hand in hands])
        np.testing.assert_array_equal(np.column_stack((cats, ties)), expected)
        np.testing.assert_array_equal(ranks, expected @ (15 ** np.arange(5, -1, -1)))


def test_rank_order_matches_native_openspiel_showdowns():
    rng = np.random.default_rng(413)
    for _ in range(300):
        deal = tuple(map(int, rng.choice(52, 9, replace=False)))
        state = state_at_street(3, holes=deal[:4], board=deal[4:])
        state.apply_action(1)
        state.apply_action(1)
        ranks, _, _ = rank_cards([(*deal[:2], *deal[4:]), (*deal[2:4], *deal[4:])])
        assert np.sign(ranks[0] - ranks[1]) == np.sign(state.returns()[0])


@pytest.mark.parametrize("text,expected", [
    ("Ac 2c 3c 4c 5c Kd Qs", (8, 5, 0, 0, 0, 0)),
    ("Ac Ad Ah Kc Kd Kh 2s", (6, 14, 13, 0, 0, 0)),
    ("Ac Ad Kc Kd Qc Qd Js", (2, 14, 13, 12, 0, 0)),
    ("2c 2d 2h 2s Ac Kc Qc", (7, 2, 14, 0, 0, 0)),
    ("Ac Kc Qc 9c 8c Jd Th", (5, 14, 13, 12, 9, 8)),
])
def test_evaluator_rare_cases(text, expected):
    _, category, ties = rank_cards([cards(text)])
    assert (category[0], *ties[0]) == expected


@pytest.mark.parametrize("hole,board", [
    ("As Ah", "Ad Kc Qh"),
    ("5c 6c", "2c 3c 4d 7h"),
    ("As 2s", "Ks Qs Js Ts 9h"),
])
def test_strength_and_blockers_match_full_independent_enumeration(hole, board):
    own, public = cards(hole), cards(board)
    actual = features(hole, board)
    rank = best_oracle((*own, *public))
    board_hands = list(combinations([c for c in range(52) if c not in public], 2))
    ranks = [best_oracle((*h, *public)) for h in board_hands]
    legal = [not set(h) & set(own) for h in board_hands]
    opponents = [r for r, valid in zip(ranks, legal) if valid]
    wins, ties = sum(r < rank for r in opponents), sum(r == rank for r in opponents)
    assert actual["current_win_fraction"] == pytest.approx(wins / len(opponents))
    assert actual["current_tie_fraction"] == pytest.approx(ties / len(opponents))
    assert actual["current_pot_share"] == pytest.approx((wins + ties / 2) / len(opponents))
    nuts = max(ranks)
    all_nuts = ranks.count(nuts)
    remaining_nuts = sum(r == nuts for r in opponents)
    assert actual["board_nuts_removed_fraction"] == pytest.approx(1 - remaining_nuts / all_nuts)
    assert actual["board_nuts_legal_mass"] == pytest.approx(remaining_nuts / len(opponents))


def test_board_only_nuts_and_preflop_missing_strength():
    f = features("2c 3d", "As Ks Qs Js Ts")
    assert f["current_tie_fraction"] == 1 and f["current_pot_share"] == .5
    assert f["currently_unbeatable"] == f["river_plays_board"] == 1
    f = features("As Ad", "")
    assert f["pocket_pair"] == 1 and f["hole_high_rank"] == 1
    assert f["strength_available"] == f["current_pot_share"] == 0
    assert sum(f[f"made_{name}"] for name in ("pair", "high_card")) == 0


def test_draws_wheel_open_ended_backdoor_and_river():
    f = features("As 2s", "3d 4h Kc")
    assert f["next_straight_completion"] == pytest.approx(4 / 47)
    f = features("9s Ts", "8d Jh 2c")
    assert f["next_straight_completion"] == pytest.approx(8 / 47)
    f = features("Ks Qs", "Js 2d 3c")
    assert f["next_flush_completion"] == 0
    assert f["runner_runner_flush"] == pytest.approx(45 / 1081)
    f = features("As Ks", "Qs Js 2c")
    assert f["next_flush_completion"] == pytest.approx(9 / 47)
    assert f["next_straight_completion"] == pytest.approx(4 / 47)
    assert f["next_either_completion"] == pytest.approx(12 / 47)  # Ts counted once
    f = features("As Ks", "Qs Js 2c 3d")
    assert f["next_flush_completion"] == pytest.approx(9 / 46)
    assert f["runner_runner_flush"] == f["runner_runner_straight"] == 0
    f = features("As Ks", "Qs Js 2c 3d 4h")
    assert all(f[name] == 0 for name in CARD_NAMES[43:49])
    f = features("As Kd", "Qs Js 2s")
    assert f["nut_flush_blocker"] == f["board_flush_blocker_rank"] == 1


def test_all_suit_permutations_hole_order_and_bounded_caches():
    own, board = cards("As 8d"), cards("Ks Qs 8s 2c 9d")
    reference = card_features(own, board)
    for perm in permutations(range(4)):
        transform = lambda group: tuple((c // 4) * 4 + perm[c % 4] for c in group)
        np.testing.assert_array_equal(card_features(transform(own[::-1]), transform(board)), reference)
    assert not reference.flags.writeable
    assert _card_features.cache_info().maxsize == HAND_CACHE_SIZE
    assert _board_table.cache_info().maxsize == BOARD_CACHE_SIZE


def test_betting_economics_exact_chip_prices_and_off_turn_mask():
    state = state_at_street()
    view = state.decision_view()
    f = card_features(view.hole_cards, view.board_cards)
    actual = dict(zip(economics_names(view.action_menu), betting_features(view, f), strict=True))
    assert actual["call_pot_odds"] == .25
    assert actual["call_closes_street"] == 0  # SB limping preserves BB option
    assert actual["pot_incremental_risk_share"] == .625  # 250/(150+250)
    assert actual["pot_opponent_call_pot_odds"] == pytest.approx(1 / 3)
    assert actual["pot_called_stack_pot_share"] == pytest.approx(19700 / 20300)
    assert actual["all_in_called_stack_pot_share"] == 0
    assert all(actual[name] == 0 for name in actual if name.startswith("half_pot_"))  # duplicate illegal
    state.apply_action(1)
    v = state.decision_view()
    b = dict(zip(economics_names(v.action_menu), betting_features(v, card_features(v.hole_cards, v.board_cards))))
    assert b["call_closes_street"] == 1 and b["call_pot_odds"] == 0
    state = state_at_street(1)
    state.apply_action(6)
    v = state.decision_view()
    b = dict(zip(economics_names(v.action_menu), betting_features(v, card_features(v.hole_cards, v.board_cards))))
    assert b["facing_all_in"] == b["call_stack_fraction"] == b["call_closes_street"] == 1
    assert b["effective_stack_pot_share"] == b["log_spr"] == 0
    off = state.decision_view(1 - v.actor)
    b = dict(zip(economics_names(v.action_menu), betting_features(off, card_features(off.hole_cards, off.board_cards))))
    assert b["observer_is_actor"] == 0
    assert b["strength_minus_pot_odds"] == b["strength_times_call_fraction"] == b["strength_times_stack_share"] == 0


def test_critic_perspectives_and_contract_reject_old_features():
    encoder = HUNLFeatureEncoder()
    state = state_at_street(2)
    x, y = encoder.information_state(state), encoder.full_state(state)
    assert encoder.policy_size == 569 and encoder.full_state_size == 688
    start, width = encoder.full_state_layout.card_size, len(CARD_NAMES)
    for player in (0, 1):
        v = state.decision_view(player)
        np.testing.assert_array_equal(y[start + player * width:start + (player + 1) * width],
                                      card_features(v.hole_cards, v.board_cards))
    np.testing.assert_array_equal(x[encoder.policy_layout.card_size:encoder.policy_layout.card_size + width],
                                  y[start + state.current_player() * width:start + (state.current_player() + 1) * width])
    old = dict(encoder.metadata(), id="hunl_exact_public_state_v2", version=2)
    with pytest.raises(ValueError, match="Unsupported"):
        encoder_from_metadata(old)
