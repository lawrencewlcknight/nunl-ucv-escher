"""Strict public-observation adapters for Slumbot and GTO Wizard.

Rebuild each response from public history with two arbitrary, disjoint opponent
cards. Only the resulting DecisionView escapes; no dummy-dependent showdown
values or critic inputs are used. Rebuilding avoids future board collisions.
"""
from dataclasses import dataclass
import math
import re

from .game import DEFAULT_ACTION_MENU, DecisionView, STACK_CHIPS, load_hunl_game


class ProtocolError(ValueError):
    """Response/history disagrees with the supported HUNL contract."""


def chips(value, name="chips"):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or int(value) != value:
        raise ProtocolError(f"{name} must be an integer chip amount")
    return int(value)


def metric(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ProtocolError(f"Invalid {name}")
    return float(value)


def cards(value):
    if isinstance(value, str):
        if len(value) % 2:
            raise ProtocolError("Invalid card string")
        value = [value[i:i + 2] for i in range(0, len(value), 2)]
    if not isinstance(value, (list, tuple)):
        raise ProtocolError("Cards must be a string or sequence")
    result = []
    for card in value:
        if not isinstance(card, str) or not re.fullmatch(r"[2-9TJQKA][cdhs]", card):
            raise ProtocolError("Invalid card notation")
        result.append(4 * "23456789TJQKA".index(card[0]) + "cdhs".index(card[1]))
    if len(set(result)) != len(result):
        raise ProtocolError("Duplicate cards")
    return tuple(result)


def history_tokens(value, *, gtow=False):
    if gtow:
        if not isinstance(value, list) or any(not isinstance(t, str) or not re.fullmatch(r"f|k|c|b[0-9]+|_", t) for t in value):
            raise ProtocolError("Malformed GTO Wizard history")
        return tuple("/" if t == "_" else t for t in value)
    if not isinstance(value, str):
        raise ProtocolError("Malformed Slumbot history")
    tokens = tuple(re.findall(r"f|k|c|b[0-9]+|/", value))
    if "".join(tokens) != value:
        raise ProtocolError("Malformed Slumbot history")
    return tokens


def reconstruct(hero, hole, board, tokens, *, action_menu=DEFAULT_ACTION_MENU, dummy_cards=None):
    if type(hero) is not int or hero not in (0, 1) or len(hole) != 2 or len(board) not in (0, 3, 4, 5):
        raise ProtocolError("Invalid seat or visible card counts")
    visible = (*hole, *board)
    if len(set(visible)) != len(visible):
        raise ProtocolError("Visible cards overlap")
    remaining = tuple(c for c in range(52) if c not in visible)
    dummy = remaining[:2] if dummy_cards is None else tuple(dummy_cards)
    if len(dummy) != 2 or len(set(dummy)) != 2 or any(c not in remaining for c in dummy):
        raise ProtocolError("Invalid temporary opponent cards")
    state = load_hunl_game(action_menu).new_initial_state()
    for card in (hole + dummy if hero == 0 else dummy + hole):
        state.apply_native_action(card)
    consumed = 0
    round_index = 0
    def deal_to(count):
        nonlocal consumed
        if len(board) < count:
            raise ProtocolError("History requires missing board cards")
        while consumed < count:
            if not state.is_chance_node():
                raise ProtocolError("Unexpected public cards")
            state.apply_native_action(board[consumed])
            consumed += 1
    try:
        for token in tokens:
            if token == "/":
                round_index += 1
                if round_index > 3:
                    if round_index != 4 or not state.is_terminal():
                        raise ProtocolError("Too many round separators")
                    continue  # optional trailing river separator
                if state.is_terminal() or not state.is_chance_node():
                    raise ProtocolError("Premature round separator")
                deal_to((0, 3, 4, 5)[round_index])
                continue
            if state.is_terminal() or state.is_chance_node():
                raise ProtocolError("Action after termination or missing street separator")
            view = state.decision_view()
            if token == "k":
                if view.to_call:
                    raise ProtocolError("Check while facing a bet")
                state.apply_native_action(1)
            elif token == "c":
                if not view.to_call:
                    raise ProtocolError("Call with nothing to call")
                state.apply_native_action(1)
            elif token == "f":
                state.apply_native_action(0)
            else:
                state.apply_street_raise_to(int(token[1:]))
        # Both APIs may omit the final separator after a completed round.
        deal_to(len(board))
    except ValueError as exc:
        if isinstance(exc, ProtocolError):
            raise
        raise ProtocolError("History contains an illegal poker action") from exc
    return state


@dataclass(frozen=True)
class RemoteHand:
    provider: str
    hand_id: int
    hero: int
    hole: tuple[int, ...]
    board: tuple[int, ...]
    history: tuple[str, ...]
    view: DecisionView | None
    winnings: float | None
    adjusted: dict[str, float]

    @property
    def terminal(self):
        return self.winnings is not None

    def action(self, index):
        if self.view is None or type(index) is not int or not 0 <= index < len(self.view.legal_mask) or not self.view.legal_mask[index]:
            raise ProtocolError("Cannot send an illegal action")
        native = self.view.native_actions[index]
        if native == 0:
            return "f", None
        if native == 1:
            return ("c" if self.view.to_call else "k"), None
        prior_streets = self.view.contributions[self.hero] - self.view.street_contributions[self.hero]
        return "b", native - prior_streets

    def log_record(self):
        # Credentials/tokens and any opponent cards are deliberately absent.
        return dict(provider=self.provider, hand_id=self.hand_id, hero=self.hero,
                    hole=list(self.hole), board=list(self.board), history=list(self.history),
                    terminal=self.terminal, winnings_chips=self.winnings,
                    winnings_bb=None if self.winnings is None else self.winnings / 100,
                    adjusted_scores=self.adjusted)


def _finish(provider, hand_id, hero, hole, board, tokens, winnings, adjusted, menu):
    state = reconstruct(hero, hole, board, tokens, action_menu=menu)
    if (winnings is not None) != state.is_terminal():
        raise ProtocolError("Settlement and reconstructed terminal state disagree")
    if winnings is not None:
        winnings = metric(winnings, "winnings")
        if abs(winnings) > STACK_CHIPS:
            raise ProtocolError("Winnings outside the supported stack size")
        # Fold settlement is independent of placeholder cards. Showdowns are not.
        if "f" in tokens and winnings != state.returns()[hero]:
            raise ProtocolError("Fold settlement disagrees with chip accounting")
        view = None
    else:
        if state.current_player() != hero:
            raise ProtocolError("API response is not our decision")
        view = state.decision_view()
    return RemoteHand(provider, hand_id, hero, hole, board, tokens, view, winnings, adjusted), state


def parse_slumbot(payload, hand_id, action_menu=DEFAULT_ACTION_MENU):
    try:
        if payload.get("error_msg"):
            raise ProtocolError("Slumbot returned an error")
        position = payload["client_pos"]
        if type(position) is not int or position not in (0, 1):
            raise ProtocolError("Invalid Slumbot position")
        adjusted = {k: metric(payload[k], k) for k in ("baseline_winnings",) if payload.get(k) is not None}
        result, _ = _finish("slumbot", hand_id, 1 - position,
            cards(payload["hole_cards"]), cards(payload["board"]), history_tokens(payload["action"]),
            payload.get("winnings"), adjusted, action_menu)
        return result
    except (KeyError, TypeError, AttributeError) as exc:
        raise ProtocolError("Missing/malformed Slumbot response fields") from exc


def _seat(position):
    if not isinstance(position, str):
        raise ProtocolError("Invalid GTO Wizard position")
    name = position.lower().replace("-", "_").replace(" ", "_")
    if name in ("sb", "btn", "button", "small_blind"):
        return 0
    if name in ("bb", "big_blind"):
        return 1
    raise ProtocolError("Unsupported GTO Wizard position")


def parse_gtow(payload, action_menu=DEFAULT_ACTION_MENU, *, hero=None):
    try:
        hand_id = chips(payload["hand_id"], "hand_id")
        game, data = payload["game"], payload["game_state"]
        if (game["game_name"] != "HUNL 200BB" or chips(game["starting_stack"]) != STACK_CHIPS
                or sorted(chips(b) for b in game["blinds"]) != [50, 100]
                or game["stack_reset_per_hand"] is not True):
            raise ProtocolError("Unsupported GTO Wizard game rules")
        players = {_seat(p["position"]): p for p in data["players"]}
        if len(data["players"]) != 2 or set(players) != {0, 1}:
            raise ProtocolError("Expected two distinct seats")
        if hero is None:
            candidates = [seat for seat, p in players.items() if p.get("hole_cards")]
            if len(candidates) != 1:
                candidates = [seat for seat, p in players.items() if p.get("name") == "user"]
            if len(candidates) != 1:
                raise ProtocolError("Cannot identify our seat without prior hand context")
            hero = candidates[0]
        winnings = data["winnings"]
        if type(data["is_hand_over"]) is not bool or data["is_hand_over"] != (winnings is not None):
            raise ProtocolError("Inconsistent GTO Wizard terminal marker")
        adjusted = {"aivat_score_chips": metric(data["aivat_score"], "aivat_score")} if data.get("aivat_score") is not None else {}
        result, state = _finish("gtow", hand_id, hero, cards(players[hero]["hole_cards"]),
            cards(data["board_cards"]), history_tokens(data["action_history"], gtow=True),
            winnings, adjusted, action_menu)
        spent = state._spent()
        if tuple(chips(players[p]["stack"]) for p in (0, 1)) != tuple(STACK_CHIPS - v for v in spent):
            raise ProtocolError("GTO Wizard stacks disagree with public history")
        if chips(data["total_pot"]) != sum(spent):
            raise ProtocolError("GTO Wizard pot disagrees with public history")
        expected_street = ("preflop", "flop", "turn", "river")[(0, 3, 4, 5).index(len(result.board))]
        if data["street"] != expected_street:
            raise ProtocolError("GTO Wizard street and board disagree")
        if result.view is not None:
            v = result.view
            if data.get("player_to_act", "user") != "user":
                raise ProtocolError("GTO Wizard says it is not our turn")
            legal = set(data["legal_actions"])
            expected = {"c" if v.to_call else "k"}
            if v.to_call:
                expected.add("f")
            if v.min_raise_to is not None:
                expected.add("b")
            if legal != expected:
                raise ProtocolError("GTO Wizard legal actions disagree with OpenSpiel")
            bounds = data["raise_range"]
            base = v.contributions[hero] - v.street_contributions[hero]
            if v.min_raise_to is None:
                if bounds is not None:
                    raise ProtocolError("Unexpected raise range")
            elif bounds is None or (chips(bounds["min"]), chips(bounds["max"])) != (v.min_raise_to - base, v.max_raise_to - base):
                raise ProtocolError("GTO Wizard raise range disagrees with OpenSpiel")
        return result
    except (KeyError, TypeError, AttributeError, IndexError) as exc:
        raise ProtocolError("Missing/malformed GTO Wizard response fields") from exc


def check_successor(before, after, sent=None):
    if before.provider != after.provider or before.hand_id != after.hand_id or before.hero != after.hero or before.hole != after.hole:
        raise ProtocolError("Hand identity changed unexpectedly")
    if after.board[:len(before.board)] != before.board or after.history[:len(before.history)] != before.history:
        raise ProtocolError("History/board is not an extension of the previous response")
    if sent is not None:
        kind, amount = sent
        expected = kind if amount is None else f"b{amount}"
        suffix = after.history[len(before.history):]
        # A response may add an omitted street separator before our action.
        suffix = tuple(t for t in suffix if t != "/")
        if not suffix or suffix[0] != expected:
            raise ProtocolError("Response does not acknowledge the submitted action")
