"""Versioned, reproducible public positions; no hidden cards in the library."""
from itertools import combinations, permutations
import hashlib
import json
from pathlib import Path

import numpy as np

from ..game import DEFAULT_ACTION_MENU, serialisable_game_definition
from ..remote import cards, history_tokens, reconstruct
from ..offline import card_text
from ..strategic_features import rank_cards, CATEGORIES

VERSION = "hunl_strategic_positions_v1"
# Assessment families are distinct public boards, not extra suit permutations.
BOARDS = (
    ("dry_ace", "As7d2c9h3s"),
    ("connected", "9sTdJc2h3s"),
    ("three_flush", "AsTs4s9d2c"),
    ("four_flush", "KsJs8s3s2d"),
    ("paired", "KhKd7c3s2h"),
    ("double_paired", "QhQd7c7s2h"),
    ("trips", "8h8d8cKs2h"),
    ("board_royal", "AsKsQsJsTs"),
    ("assessment_dry", "Kh8c2dTs4h"),
    ("assessment_flush", "Qh9h3h6h2c"),
    ("assessment_paired", "JhJc9d5s2h"),
    ("assessment_straight", "5c6d7h8s9c"),
)
RIVER_HISTORIES = (
    ("limped_oop", 1, "ck/kk/kk/"),
    ("checked_to_ip", 0, "b300c/kk/kk/k"),
    ("half_pot", 0, "b300c/kk/kk/b300"),
    ("pot", 1, "b300c/kk/kk/kb600"),
    ("overbet", 0, "b300c/kk/kk/b1200"),
    ("all_in", 1, "b300c/kk/kk/kb19700"),
    ("three_bet_pot", 0, "b300b1000c/kk/kk/b1000"),
    ("three_barrels", 0, "b300c/b300c/b900c/b2500"),
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def board_class(board):
    return min(tuple(sorted(4*(c//4)+p[c%4] for c in board))
               for p in permutations(range(4)))


def state_for(position, menu=DEFAULT_ACTION_MENU, dummy=None):
    state = reconstruct(position["hero"], cards(position["hole"]), cards(position["board"]),
                        history_tokens(position["history"]), action_menu=menu, dummy_cards=dummy)
    if state.is_terminal() or state.is_chance_node() or state.current_player() != position["hero"]:
        raise ValueError(f"Position {position['id']} is not the hero's decision")
    return state


def make_library(seed=20261118, smoke=False):
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    positions = []
    for bi, (family, text) in enumerate(BOARDS):
        if smoke and bi not in (0, 7, 8):
            continue
        # Randomize suit names once; every checkpoint sees exactly the same cards.
        perm = rng.permutation(4)
        board = tuple(4*(c//4)+int(perm[c%4]) for c in cards(text))
        hands = np.array(list(combinations([c for c in range(52) if c not in board], 2)))
        ranks, cats, _ = rank_cards(np.column_stack((hands, np.tile(board, (len(hands), 1)))))
        # One representative per present made-hand category plus a strongest hand.
        selected = [int(rng.choice(np.flatnonzero(cats == c))) for c in np.unique(cats)]
        selected += [int(np.argmax(ranks))]
        selected = list(dict.fromkeys(selected))
        if smoke:
            selected = list(dict.fromkeys([selected[0], selected[-1]]))
        for hi in selected:
            for label, hero, history in (RIVER_HISTORIES if not smoke else
                                        (RIVER_HISTORIES[1], RIVER_HISTORIES[5])):
                positions.append(dict(id=f"{family}-{hi}-{label}", family=family,
                    partition="assessment" if bi >= 8 else "development", hero=hero,
                    hole="".join(map(card_text, hands[hi])), board="".join(map(card_text, board)),
                    history=history, tags=[family, CATEGORIES[int(cats[hi])], label, "river"]))
    # Earlier streets are descriptive probes, never assigned guessed optimal actions.
    early = (
        ("preflop_premium", "AsAh", "", 0, ""),
        ("preflop_suited", "KsQs", "", 1, "b300"),
        ("preflop_low_pair", "3c3d", "", 0, "b300b1000"),
        ("preflop_weak", "7c2d", "", 0, ""),
        ("flop_nut_draw", "AsQs", "Js7s2d", 0, "b300c/b300"),
        ("flop_combo_draw", "9sTs", "8sJd2s", 0, "b300c/b300"),
        ("flop_overpair", "AcAd", "9s6d2c", 1, "b300c/"),
        ("flop_air", "5h4h", "AsKd9c", 0, "b300c/b600"),
        ("turn_nut_draw", "AsQs", "Js7s2d4c", 0, "b300c/kk/b600"),
        ("turn_made_flush", "AsQs", "Js7s2d4s", 0, "b300c/kk/b600"),
        ("turn_straight", "9sTs", "8cJd2sQh", 1, "b300c/kk/"),
        ("turn_all_in_nut_draw", "AsQs", "Js7s2d4c", 0, "b300c/kk/b19700"),
        ("turn_all_in_made_flush", "AsQs", "Js7s2d4s", 0, "b300c/kk/b19700"),
        ("turn_bluff_catcher", "AcTd", "AsKd9c8s", 0, "b300c/b300c/b1200"),
    )
    for name, hole, board, hero, history in early:
        positions.append(dict(id=name, family=name, partition="development", hero=hero,
            hole=hole, board=board, history=history, tags=[name, "exact_all_in" if "all_in" in name else "behaviour_only"]))
    if not smoke:
        # Explicit blocker/kicker contrasts share board and betting history.
        # They are observations, not prescribed direction-of-action tests.
        contrasts = (
            ("flush_blocker", "AsTs4s9d2c", ("KsQh", "KcQh")),
            ("pair_kicker", "Ah7c6s4d3h", ("AcQd", "AcJd", "Ac2d")),
            ("straight_blocker", "9sTdJc2h3s", ("Qs6h", "7s6h")),
        )
        for family, board, holdings in contrasts:
            for hole in holdings:
                for label, hero, history in RIVER_HISTORIES:
                    positions.append(dict(id=f"{family}-{hole}-{label}",
                        family={"flush_blocker": "three_flush", "straight_blocker": "connected"}.get(family, family),
                        partition="development", hero=hero, hole=hole, board=board,
                        history=history, tags=[family, label, "river", "contrast"]))
    result = dict(schema=VERSION, seed=seed, smoke=smoke,
                  game=serialisable_game_definition(), positions=positions)
    validate_library(result)
    return result


def validate_library(library):
    if library.get("schema") != VERSION or library.get("game") != serialisable_game_definition():
        raise ValueError("Unsupported library schema/game")
    positions = library.get("positions")
    if not isinstance(positions, list) or not positions:
        raise ValueError("Library requires positions")
    ids, split_classes, family_splits = set(), {}, {}
    for p in positions:
        if not isinstance(p.get("id"), str) or not p["id"] or p["id"] in ids:
            raise ValueError("Position IDs must be unique nonempty strings")
        ids.add(p["id"])
        if p.get("partition") not in ("development", "assessment"):
            raise ValueError("Invalid partition")
        if not isinstance(p.get("family"), str) or not p["family"]:
            raise ValueError("Position requires a family")
        if family_splits.setdefault(p["family"], p["partition"]) != p["partition"]:
            raise ValueError("A family cannot cross partitions")
        state = state_for(p)
        b = state.decision_view().board_cards
        if b:
            key = board_class(b)
            if split_classes.setdefault(key, p["partition"]) != p["partition"]:
                raise ValueError("Suit-equivalent boards cross partitions")
    return library
