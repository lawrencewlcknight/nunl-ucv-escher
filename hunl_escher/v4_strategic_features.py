"""Bounded Hold'em descriptors with cached, deterministic preflop equity.

Strength compares the best CURRENT five cards with every legal uniform opponent
holding. Before the river this is not showdown equity. Draw probabilities describe
card patterns, not clean winning outs. Neither computation sees hidden cards.
"""
from functools import lru_cache
from itertools import combinations

import numpy as np

CATEGORIES = ("high_card", "pair", "two_pair", "trips", "straight", "flush",
              "full_house", "quads", "straight_flush")
CARD_NAMES = (
    "hole_high_rank", "hole_low_rank", "pocket_pair", "hole_suited", "hole_rank_gap",
    "hole_shared_straight_window", "hole_broadway_fraction",
    *(f"made_{name}" for name in CATEGORIES),
    *(f"made_tiebreak_{i}" for i in range(5)),
    "strength_available", "current_win_fraction", "current_tie_fraction",
    "current_loss_fraction", "current_pot_share", "currently_unbeatable",
    "board_nuts_removed_fraction", "board_nuts_legal_mass",
    "board_paired_rank_fraction", "board_max_multiplicity", "board_max_suit_fraction",
    "board_straight_window_fraction", "board_high_rank", "board_low_rank",
    "river_board_category", "river_plays_board",
    "pocket_overpair", "hole_matches_board_top", "hole_overcard_fraction",
    "hole_matches_board_fraction", "board_flush_blocker_rank", "nut_flush_blocker",
    "next_straight_completion", "next_flush_completion", "next_either_completion",
    "runner_runner_straight", "runner_runner_flush", "next_category_improvement",
    "next_card_available", *(f"next_card_{name}" for name in CATEGORIES),
    "preflop_equity_available", "preflop_equity", "preflop_equity_sampling_se",
)
ECONOMICS_NAMES = (
    "observer_is_actor", "observer_in_position", "call_pot_odds", "call_stack_fraction",
    "effective_stack_pot_share", "log_spr", "facing_all_in", "call_closes_street",
    "strength_minus_pot_odds", "strength_times_call_fraction", "strength_times_stack_share",
)
ACTION_NAMES = ("incremental_risk_share", "opponent_call_pot_odds", "called_stack_pot_share")
AGGRESSION_NAMES = (*(f"street_{s}_raise_count_share" for s in range(4)),
                    *(f"street_{s}_aggressor_{who}" for s in range(4)
                      for who in ("self", "opponent")))
BOARD_CACHE_SIZE = 64
HAND_CACHE_SIZE = 8192
_PAIRS = np.asarray(list(combinations(range(52), 2)), dtype=np.int16)
_RANK_BITS = 1 << np.arange(13)
_POWERS = 15 ** np.arange(4, -1, -1)
_WINDOWS = (0x100F, *(31 << start for start in range(9)))
_STRAIGHT = np.zeros(8192, dtype=np.int8)
for _mask in range(8192):
    if _mask & 0x100F == 0x100F:
        _STRAIGHT[_mask] = 5
    for _start in range(9):
        if _mask & (31 << _start) == 31 << _start:
            _STRAIGHT[_mask] = _start + 6


def rank_cards(cards):
    """Vectorised exact best-five evaluator for N x (5, 6 or 7) rank*4+suit IDs.

    Returns a sortable integer, category 0..8, and five lexicographic kickers
    (2..14, zero padding). Handles two trips, three pairs and wheel flushes.
    """
    cards = np.asarray(cards, dtype=np.int16)
    if cards.ndim != 2 or cards.shape[1] not in (5, 6, 7):
        raise ValueError("Expected N by 5, 6 or 7 cards")
    ranks, suits = cards // 4, cards % 4
    counts = np.eye(13, dtype=np.int8)[ranks].sum(axis=1)
    mask = ((counts > 0) * _RANK_BITS).sum(axis=1)
    straight = _STRAIGHT[mask]
    category = np.zeros(len(cards), dtype=np.int8)
    rank_values = np.arange(2, 15)

    def top(where, n=5):
        return np.sort(np.where(where, rank_values, 0), axis=1)[:, -n:][:, ::-1]

    tie = top(counts > 0)
    pairs, trips, quads = top(counts >= 2, 3), top(counts >= 3, 2), top(counts == 4, 1)
    pair = pairs[:, 0] > 0
    category[pair] = 1
    pair_tie = np.column_stack((pairs[:, 0], top((counts > 0) & (rank_values != pairs[:, :1]), 3),
                               np.zeros(len(cards), dtype=int)))
    tie[pair] = pair_tie[pair]
    two_pair = pairs[:, 1] > 0
    category[two_pair] = 2
    two_tie = np.column_stack((pairs[:, :2], top((counts > 0) &
        (rank_values != pairs[:, :1]) & (rank_values != pairs[:, 1:2]), 1),
        np.zeros((len(cards), 2), dtype=int)))
    tie[two_pair] = two_tie[two_pair]
    trip = trips[:, 0] > 0
    category[trip] = 3
    trip_tie = np.column_stack((trips[:, 0], top((counts > 0) &
        (rank_values != trips[:, :1]), 2), np.zeros((len(cards), 2), dtype=int)))
    tie[trip] = trip_tie[trip]
    category[straight > 0] = 4
    tie[straight > 0] = 0
    tie[straight > 0, 0] = straight[straight > 0]
    flush_straight = np.zeros(len(cards), dtype=np.int8)
    for suit in range(4):
        suited = suits == suit
        flush = suited.sum(axis=1) >= 5
        flush_ranks = np.zeros_like(counts, dtype=bool)
        rows, cols = np.nonzero(suited)
        flush_ranks[rows, ranks[rows, cols]] = True
        category[flush] = 5
        tie[flush] = top(flush_ranks)[flush]
        flush_straight = np.maximum(flush_straight,
            _STRAIGHT[(flush_ranks * _RANK_BITS).sum(axis=1)])
    other_pair = top((counts >= 2) & (rank_values != trips[:, :1]), 1)[:, 0]
    full = trip & (other_pair > 0)
    category[full] = 6
    tie[full] = 0
    tie[full, 0], tie[full, 1] = trips[full, 0], other_pair[full]
    quad = quads[:, 0] > 0
    category[quad] = 7
    tie[quad] = 0
    tie[quad, 0] = quads[quad, 0]
    tie[quad, 1] = top((counts > 0) & (rank_values != quads), 1)[quad, 0]
    sf = flush_straight > 0
    category[sf] = 8
    tie[sf] = 0
    tie[sf, 0] = flush_straight[sf]
    return category.astype(np.int64) * 15**5 + tie @ _POWERS, category, tie


@lru_cache(maxsize=BOARD_CACHE_SIZE)
def _board_table(board):
    hands = _PAIRS[~np.isin(_PAIRS, board).any(axis=1)]
    ranks, categories, ties = rank_cards(np.column_stack((hands, np.tile(board, (len(hands), 1)))))
    lookup = np.full((52, 52), -1, dtype=np.int16)
    lookup[hands[:, 0], hands[:, 1]] = np.arange(len(hands))
    nuts = ranks == ranks.max()
    for value in (hands, ranks, categories, ties, lookup, nuts):
        value.flags.writeable = False
    return hands, ranks, categories, ties, lookup, nuts


def _canonical_cards(hole, board):
    # Board-first ordering shares expensive tables across different own holdings.
    signatures = [(sum(1 << (c // 4) for c in board if c % 4 == suit),
                   sum(1 << (c // 4) for c in hole if c % 4 == suit)) for suit in range(4)]
    order = sorted(range(4), key=lambda suit: signatures[suit], reverse=True)
    mapping = tuple(order.index(suit) for suit in range(4))
    return tuple(tuple(sorted(4 * (c // 4) + mapping[c % 4] for c in group))
                 for group in (hole, board))


def card_features(hole, board):
    """Read-only cached vector. Only own two cards and visible board are accepted."""
    cards = (*hole, *board)
    if (len(hole) != 2 or len(board) not in (0, 3, 4, 5) or len(set(cards)) != len(cards)
            or any(type(c) is not int or not 0 <= c < 52 for c in cards)):
        raise ValueError("Invalid own/public cards")
    return _card_features(*_canonical_cards(hole, board))


@lru_cache(maxsize=HAND_CACHE_SIZE)
def _card_features(hole, board):
    values = dict.fromkeys(CARD_NAMES, 0.)
    hr = [c // 4 + 2 for c in hole]
    hole_mask = sum(1 << (r - 2) for r in set(hr))
    values.update(hole_high_rank=max(hr) / 14, hole_low_rank=min(hr) / 14,
        pocket_pair=hr[0] == hr[1], hole_suited=hole[0] % 4 == hole[1] % 4,
        hole_rank_gap=abs(hr[0] - hr[1]) / 12,
        hole_shared_straight_window=hr[0] != hr[1] and any(hole_mask & w == hole_mask for w in _WINDOWS),
        hole_broadway_fraction=sum(r >= 10 for r in hr) / 2)
    if board:
        hands, ranks, categories, ties, lookup, nuts = _board_table(board)
        idx = lookup[hole]
        rank, category, tie = ranks[idx], categories[idx], ties[idx]
        legal = ~np.isin(hands, hole).any(axis=1)
        win, equal, loss = ((ranks[legal] < rank).mean(), (ranks[legal] == rank).mean(),
                            (ranks[legal] > rank).mean())
        values.update(strength_available=1, current_win_fraction=win, current_tie_fraction=equal,
            current_loss_fraction=loss, current_pot_share=win + equal / 2,
            currently_unbeatable=loss == 0,
            board_nuts_removed_fraction=1 - np.count_nonzero(nuts & legal) / nuts.sum(),
            board_nuts_legal_mass=np.count_nonzero(nuts & legal) / legal.sum())
        values[f"made_{CATEGORIES[category]}"] = 1
        values.update({f"made_tiebreak_{i}": t / 14 for i, t in enumerate(tie)})
        br = [c // 4 + 2 for c in board]
        bc = np.bincount(np.asarray(br) - 2, minlength=13)
        bs = np.bincount(np.asarray(board) % 4, minlength=4)
        board_mask = sum(1 << (r - 2) for r in set(br))
        values.update(board_paired_rank_fraction=np.count_nonzero(bc >= 2) / 2,
            board_max_multiplicity=bc.max() / 4, board_max_suit_fraction=bs.max() / 5,
            board_straight_window_fraction=max((board_mask & w).bit_count() for w in _WINDOWS) / 5,
            board_high_rank=max(br) / 14, board_low_rank=min(br) / 14,
            pocket_overpair=hr[0] == hr[1] and hr[0] > max(br),
            hole_matches_board_top=max(br) in hr, hole_overcard_fraction=sum(r > max(br) for r in hr) / 2,
            hole_matches_board_fraction=sum(r in br for r in hr) / 2)
        if len(board) == 5:
            board_rank, board_cat, _ = rank_cards([board])
            values.update(river_board_category=board_cat[0] / 8, river_plays_board=rank == board_rank[0])
        for suit in range(4):
            if bs[suit] < 3:
                continue
            own = [c // 4 + 2 for c in hole if c % 4 == suit]
            unseen_ranks = {r for r in range(2, 15) if (r - 2) * 4 + suit not in board}
            values["board_flush_blocker_rank"] = max(own, default=0) / 14
            values["nut_flush_blocker"] = max(unseen_ranks) in own
        if len(board) < 5:
            visible = (*hole, *board)
            remaining = np.asarray([c for c in range(52) if c not in visible], dtype=np.int16)
            mask = hole_mask | board_mask
            suits = np.bincount(np.asarray(visible) % 4, minlength=4)
            # Existing made patterns have no completion draw. Paired boards and
            # opponent redraws mean these are NOT probabilities of winning.
            straight = (_STRAIGHT[mask | _RANK_BITS[remaining // 4]] > 0) & (_STRAIGHT[mask] == 0)
            flush = (suits[remaining % 4] == 4) & (suits.max() < 5)
            values.update(next_straight_completion=straight.mean(), next_flush_completion=flush.mean(),
                          next_either_completion=(straight | flush).mean())
            _, next_cat, _ = rank_cards(np.column_stack((np.tile(visible, (len(remaining), 1)), remaining)))
            values["next_category_improvement"] = (next_cat > category).mean()
            values["next_card_available"] = 1
            values.update({f"next_card_{name}": mass for name, mass in
                           zip(CATEGORIES, np.bincount(next_cat, minlength=9) / len(next_cat))})
            if len(board) == 3:
                runouts = _PAIRS[~np.isin(_PAIRS, visible).any(axis=1)]
                r0, r1 = runouts[:, 0] // 4, runouts[:, 1] // 4
                both = _STRAIGHT[mask | _RANK_BITS[r0] | _RANK_BITS[r1]] > 0
                neither = (_STRAIGHT[mask | _RANK_BITS[r0]] == 0) & (_STRAIGHT[mask | _RANK_BITS[r1]] == 0)
                values["runner_runner_straight"] = (both & neither).mean()
                s0, s1 = runouts[:, 0] % 4, runouts[:, 1] % 4
                values["runner_runner_flush"] = ((s0 == s1) & (suits[s0] == 3)).mean()
    else:
        from .preflop_equity import preflop_equity
        equity, se = preflop_equity(hole)
        values.update(preflop_equity_available=1, preflop_equity=equity,
                      preflop_equity_sampling_se=se)
    result = np.asarray([values[name] for name in CARD_NAMES], dtype=np.float32)
    result.flags.writeable = False
    return result


def economics_names(action_menu):
    return (*ECONOMICS_NAMES, *(f"{action}_{name}" for action in action_menu for name in ACTION_NAMES),
            *AGGRESSION_NAMES)


@lru_cache(maxsize=4096)
def public_aggression(history):
    """Counts and absolute-seat last aggressors from redacted public events."""
    counts, aggressors = [0] * 4, [-1] * 4
    dealt = street = actor = 0
    for event in history:
        if event == 0:
            dealt += 1
            street = 0 if dealt <= 4 else min(3, max(1, dealt - 6))
            actor = 0 if street == 0 else 1
        else:
            if event > 0:
                counts[street] += 1
                aggressors[street] = actor
            actor = 1 - actor
    return tuple(counts), tuple(aggressors)


def critic_comparison(features0, features1):
    """Compare already-computed exact ranks; privileged critic channels only."""
    if not features0[CARD_NAMES.index("strength_available")]:
        return np.zeros(4, dtype=np.float32)
    def rank(features):
        category = int(np.argmax(features[7:16]))
        return (category, *features[16:21])
    left, right = rank(features0), rank(features1)
    return np.asarray((1, left > right, left == right, left < right), dtype=np.float32)


def betting_features(view, own_features):
    """Actor economics; own-strength interactions masked for off-turn observers.

    x/(pot+x) is a bounded, monotone representation of stack/pot or cost/pot.
    Action counterfactuals assume the opponent calls, never predict their response.
    """
    actor, opponent = view.actor, 1 - view.actor
    cost = min(view.to_call, view.stacks[actor])
    called_pot = view.pot + cost
    effective = min(view.stacks[actor] - cost, view.stacks[opponent])
    odds = cost / called_pot
    risk = cost / max(view.stacks[actor], 1)
    share = effective / (called_pot + effective)
    acting = view.player == actor
    strength = float(own_features[CARD_NAMES.index("current_pot_share")])
    available = bool(own_features[CARD_NAMES.index("strength_available")]) and acting
    # Last zero is the final dealt card; private cards are also redacted zeros.
    last_deal = max(i for i, event in enumerate(view.public_history) if event == 0)
    closes = len(view.public_history) > last_deal + 1
    result = [acting, view.player == 0, odds, risk, share,
              np.log1p(effective / called_pot) / np.log(201),
              view.stacks[opponent] == 0 and cost > 0, closes,
              strength - odds if available else 0,
              strength * risk if available else 0, strength * share if available else 0]
    for action in view.native_actions:
        if action is None or action == 0:
            result.extend((0, 0, 0))
            continue
        delta = cost if action == 1 else action - view.contributions[actor]
        response = 0 if action == 1 else min(view.stacks[opponent], action - view.contributions[opponent])
        pot_after_response = view.pot + delta + response
        left = min(view.stacks[actor] - delta, view.stacks[opponent] - response)
        result.extend((delta / (view.pot + delta), response / pot_after_response,
                       left / (pot_after_response + left)))
    counts, aggressors = public_aggression(tuple(view.public_history))
    result.extend(n / (n + 1) for n in counts)
    result.extend(a == p for a in aggressors for p in (view.player, 1 - view.player))
    return np.asarray(result, dtype=np.float32)
