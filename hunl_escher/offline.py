"""Synthetic protocol fixtures and equivalence checks; makes no HTTP requests."""
import numpy as np

from .features import HUNLFeatureEncoder
from .game import DEFAULT_ACTION_MENU, STACK_CHIPS, load_hunl_game
from .remote import parse_gtow, parse_slumbot


def card_text(card):
    return "23456789TJQKA"[card // 4] + "cdhs"[card % 4]


def synthetic_payloads(state, hero, *, hand_id=1):
    """Serialize independent native events into each documented wire format.

    These fixtures are synthetic, not captured server responses. Betting history
    is derived from native event IDs, not from either adapter's parser.
    """
    chance_cards, tokens = [], []
    spent, street_start = [50, 100], [0, 0]
    for event in state.native_history():
        action, actor = event.action, event.player
        if actor == -1:
            if len(chance_cards) in (4, 7, 8):
                tokens.append("/")
                street_start = list(spent)
            chance_cards.append(action)
        elif action == 0:
            tokens.append("f")
        elif action == 1:
            tokens.append("c" if spent[actor] < max(spent) else "k")
            spent[actor] = max(spent)
        else:
            tokens.append(f"b{action - street_start[actor]}")
            spent[actor] = action
    hole = chance_cards[2 * hero:2 * hero + 2]
    board = chance_cards[4:]
    winnings = state.returns()[hero] if state.is_terminal() else None
    slumbot = dict(token="SYNTHETIC_ONLY", client_pos=1 - hero,
        hole_cards=[card_text(c) for c in hole], board=[card_text(c) for c in board],
        action="".join(tokens))
    if winnings is not None:
        slumbot["winnings"] = winnings
        slumbot["baseline_winnings"] = winnings + 7.25  # intentionally distinct metric
    view = None if state.is_terminal() else state.decision_view(hero)
    base = spent[hero] - (0 if view is None else view.street_contributions[hero])
    legal = []
    bounds = None
    if view is not None:
        legal = ["c" if view.to_call else "k"]
        if view.to_call:
            legal.append("f")
        if view.min_raise_to is not None:
            legal.append("b")
            bounds = dict(min=view.min_raise_to - base, max=view.max_raise_to - base)
    players = [dict(name="user" if p == hero else "gto_wizard", position="SB" if p == 0 else "BB",
        stack=STACK_CHIPS - spent[p], hole_cards="".join(map(card_text, hole)) if p == hero else None) for p in (0, 1)]
    gtow = dict(hand_id=hand_id, game=dict(game_id=1, game_name="HUNL 200BB", game_format="HUNL",
        starting_stack=STACK_CHIPS, blinds=[50, 100], stack_reset_per_hand=True),
        game_state=dict(street=("preflop", "flop", "turn", "river")[(0, 3, 4, 5).index(len(board))],
            common_pot=sum(street_start), total_pot=sum(spent), board_cards="".join(map(card_text, board)),
            is_hand_over=state.is_terminal(), player_to_act=None if state.is_terminal() else "user",
            players=players, legal_actions=legal, raise_range=bounds,
            action_history=["_" if t == "/" else t for t in tokens],
            has_gto_wizard_folded=state.is_terminal() and bool(tokens) and tokens[-1] == "f" and state.native_history()[-1].player != hero,
            winnings=winnings, aivat_score=None if winnings is None else winnings + 13.5))
    return slumbot, gtow


def run_equivalence(hands=100, *, seed=0, action_menu=DEFAULT_ACTION_MENU, policy=None):
    """Every decision through both parsers, including arbitrary off-menu raises."""
    if hands < 1:
        raise ValueError("hands must be positive")
    game, encoder = load_hunl_game(action_menu), HUNLFeatureEncoder(action_menu)
    rng = np.random.default_rng(seed)
    counts = dict(hands=hands, decisions=0, adapter_observations=0, off_menu_raises=0,
                  policy_probability_checks=0, terminal_checks=0)
    streets, seats = set(), set()
    for hand_id in range(1, hands + 1):
        state = game.new_initial_state()
        while not state.is_terminal():
            if state.is_chance_node():
                state.apply_action(int(rng.choice(state.legal_actions())))
                continue
            hero = state.current_player()
            source = state.decision_view()
            payloads = synthetic_payloads(state, hero, hand_id=hand_id)
            parsed = (parse_slumbot(payloads[0], hand_id, action_menu), parse_gtow(payloads[1], action_menu))
            expected = encoder.information_state(state)
            for result in parsed:
                assert result.view == source, "Public view mismatch"
                np.testing.assert_array_equal(encoder.encode_decision(result.view), expected)
                if policy is not None:
                    assert policy.action_probabilities_from_view(result.view) == policy.action_probabilities(state)
                    counts["policy_probability_checks"] += 1
                for index in state.legal_actions():
                    kind, amount = result.action(index)
                    actual = state.native_action(index)
                    if kind == "b":
                        base = source.contributions[hero] - source.street_contributions[hero]
                        assert amount + base == actual
                    else:
                        assert actual == (0 if kind == "f" else 1)
                counts["adapter_observations"] += 1
            seats.add(hero)
            streets.add(source.street)
            counts["decisions"] += 1
            # Half the hands predominantly check/call, to cover later streets.
            roll = rng.random()
            if hand_id % 2 == 0 and roll < .9:
                state.apply_action(1)
            elif source.min_raise_to is not None and roll < .55:
                native = int(rng.integers(source.min_raise_to, source.max_raise_to + 1))
                counts["off_menu_raises"] += int(native not in source.native_actions)
                state.apply_native_action(native)
            else:
                state.apply_action(int(rng.choice(state.legal_actions())))
        for hero in (0, 1):
            a, b = synthetic_payloads(state, hero, hand_id=hand_id)
            for result in (parse_slumbot(a, hand_id, action_menu), parse_gtow(b, action_menu)):
                assert result.terminal and result.winnings == state.returns()[hero]
                counts["terminal_checks"] += 1
    return {**counts, "streets": sorted(streets), "seats": sorted(seats),
            "kind": "synthetic_offline_equivalence", "seed": seed}
