"""Exact OpenSpiel HUNL engine with a small learner action space.

Native raises are total contributions over the HAND; apply_street_raise_to
accepts contributions over the STREET. Learner IDs are menu indices.
"""
from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType

import pyspiel

HUNL_GAME_NAME = "HUNL"
OPEN_SPIEL_GAME_NAME = "universal_poker"
STACK_CHIPS = 20000
INTERFACE_VERSION = 2
DEFAULT_ACTION_MENU = ("fold", "check_call", "min_raise", "half_pot", "pot",
                       "double_pot", "all_in")
POT_FRACTIONS = {"half_pot": Fraction(1, 2), "pot": Fraction(1),
                 "double_pot": Fraction(2)}
HUNL_GAME_PARAMETERS = MappingProxyType({
    "betting": "nolimit", "bettingAbstraction": "fullgame",
    "blind": "50 100", "stack": "20000 20000",
    "firstPlayer": "1 2 2 2", "numPlayers": 2, "numRounds": 4,
    "numSuits": 4, "numRanks": 13, "numHoleCards": 2,
    "numBoardCards": "0 3 1 1",
})


def validate_action_menu(action_menu):
    menu = tuple(action_menu)
    if (menu[:2] != ("fold", "check_call") or "all_in" not in menu
            or len(set(menu)) != len(menu)
            or any(name not in DEFAULT_ACTION_MENU for name in menu)):
        raise ValueError("Action menu must start fold/check_call, include all_in, "
                         "and contain unique supported action names")
    return menu


def serialisable_game_definition(action_menu=DEFAULT_ACTION_MENU):
    return {
        "name": HUNL_GAME_NAME, "open_spiel_game": OPEN_SPIEL_GAME_NAME,
        "parameters": dict(HUNL_GAME_PARAMETERS),
        "interface_version": INTERFACE_VERSION,
        "action_menu": list(validate_action_menu(action_menu)),
        "raise_units": "total contribution over hand",
        "pot_fraction_rule": "raise by fraction of pot after calling; round half up",
        "duplicate_rule": "first menu index wins; all_in reserves maximum raise",
        "stack_reset": "every hand", "rake": 0,
    }


@dataclass(frozen=True)
class DecisionView:
    """One player's information only. P0=button/SB, P1=BB.

    player selects private cards; actor, to_call and legal actions describe the
    current decision. UCV also requests the traverser view at opponent nodes.
    """
    player: int
    actor: int
    street: int
    hole_cards: tuple[int, ...]
    board_cards: tuple[int, ...]  # first three flop, then turn, then river
    contributions: tuple[int, int]
    street_contributions: tuple[int, int]
    stacks: tuple[int, int]
    pot: int
    to_call: int
    min_raise_to: int | None
    max_raise_to: int | None
    legal_mask: tuple[int, ...]
    native_actions: tuple[int | None, ...]
    # One value/event: chance=0, fold=-1, check/call=-.5, raise=to/stack.
    # Chance CARD IDENTITIES never enter this public sequence.
    public_history: tuple[float, ...]
    action_menu: tuple[str, ...] = DEFAULT_ACTION_MENU


class HUNLGame:
    """Learner-facing Python adapter, not a registered C++ OpenSpiel game."""
    def __init__(self, action_menu=DEFAULT_ACTION_MENU):
        self.action_menu = validate_action_menu(action_menu)
        self.engine = pyspiel.load_game(OPEN_SPIEL_GAME_NAME, dict(HUNL_GAME_PARAMETERS))

    def new_initial_state(self):
        return HUNLState(self, self.engine.new_initial_state())

    def definition(self):
        return serialisable_game_definition(self.action_menu)

    def num_distinct_actions(self):
        return len(self.action_menu)

    def num_players(self):
        return self.engine.num_players()

    def max_game_length(self):
        return self.engine.max_game_length()

    def max_utility(self):
        return self.engine.max_utility()

    def min_utility(self):
        return self.engine.min_utility()

    def information_state_tensor_size(self):
        # Initial discovery/diagnostics only. Neural inputs use features.py.
        return self.engine.information_state_tensor_size()


class HUNLState:
    def __init__(self, game, native):
        self._game, self._native = game, native
        self._cards = ()
        self._public_history = ()
        self._street_start = (0, 0)
        self._bounds_cache = self._spent_cache = self._menu_cache = None
        # Encoder-owned entries live only as long as this unchanged state.
        # Clones start empty so parent and child caches never share ownership.
        self._feature_cache = {}

    def get_game(self):
        return self._game

    def current_player(self):
        return self._native.current_player()

    def is_chance_node(self):
        return self._native.is_chance_node()

    def is_terminal(self):
        return self._native.is_terminal()

    def num_distinct_actions(self):
        return self._game.num_distinct_actions()

    def chance_outcomes(self):
        return self._native.chance_outcomes()

    def returns(self):
        return self._native.returns()

    def information_state_tensor(self, player=None):
        """Native diagnostic tensor; native actions are NOT learner menu IDs."""
        return self._native.information_state_tensor(
            self.current_player() if player is None else player)

    def information_state_string(self, player=None):
        return self._native.information_state_string(
            self.current_player() if player is None else player)

    def native_history(self):
        """Diagnostic history in OpenSpiel IDs, including private chance cards."""
        return self._native.full_history()

    def clone(self):
        result = HUNLState(self._game, self._native.clone())
        result._cards = self._cards
        result._public_history = self._public_history
        result._street_start = self._street_start
        result._bounds_cache = self._bounds_cache
        result._spent_cache = self._spent_cache
        result._menu_cache = self._menu_cache
        return result

    def child(self, action):
        result = self.clone()
        result.apply_action(action)
        return result

    def _spent(self):
        if self._spent_cache is None:
            # Pinned observation schema: player2 + hole52 + board52 + spent2.
            observation = self._native.observation_tensor(0)
            if len(observation) != 108:
                raise ValueError("OpenSpiel HUNL observation schema changed")
            self._spent_cache = tuple(int(x) for x in observation[106:108])
        return self._spent_cache

    def _bounds(self):
        if self._bounds_cache is None:
            actions = self._native.legal_actions()
            first_raise = next((a for a in actions if a >= 2), None)
            self._bounds_cache = (0 in actions, 1 in actions, first_raise,
                                  actions[-1] if first_raise is not None else None)
        return self._bounds_cache

    def _menu(self):
        if self._menu_cache is not None:
            return self._menu_cache
        if self.is_terminal() or self.is_chance_node():
            return (None,) * self.num_distinct_actions()
        fold, call, minimum, maximum = self._bounds()
        spent = self._spent()
        to_call = max(spent) - spent[self.current_player()]
        pot_after_call = sum(spent) + to_call
        result, seen = [], set()
        for name in self._game.action_menu:
            candidate = None
            if name == "fold" and fold:
                candidate = 0
            elif name == "check_call" and call:
                candidate = 1
            elif minimum is not None:
                if name == "all_in":
                    candidate = maximum
                elif name == "min_raise":
                    candidate = minimum
                elif name in POT_FRACTIONS:
                    size = POT_FRACTIONS[name] * pot_after_call
                    rounded = (2 * size.numerator + size.denominator) // (2 * size.denominator)
                    candidate = max(spent) + rounded
                if candidate is not None and not minimum <= candidate <= maximum:
                    candidate = None
                if candidate == maximum and name != "all_in":
                    candidate = None
            if candidate in seen:
                candidate = None
            if candidate is not None:
                seen.add(candidate)
            result.append(candidate)
        self._menu_cache = tuple(result)
        return self._menu_cache

    def legal_actions(self, player=None):
        if player is not None and player != self.current_player():
            return []
        if self.is_chance_node():
            return self._native.legal_actions()
        return [i for i, action in enumerate(self._menu()) if action is not None]

    def legal_actions_mask(self, player=None):
        if self.is_chance_node():
            raise ValueError("Learner masks are defined only at decision/terminal states")
        legal = set(self.legal_actions(player))
        return [int(i in legal) for i in range(self.num_distinct_actions())]

    def native_action(self, action):
        if self.is_chance_node() or self.is_terminal():
            raise ValueError("Menu translation requires a decision state")
        if type(action) is not int or not 0 <= action < self.num_distinct_actions():
            raise ValueError("Invalid learner action index")
        native = self._menu()[action]
        if native is None:
            raise ValueError("Illegal learner action")
        return native

    def apply_action(self, action):
        if self.is_chance_node():
            self.apply_native_action(action)
        else:
            self.apply_native_action(self.native_action(action))

    def apply_native_action(self, action):
        """Exact 0=fold, 1=check/call, N=raise-to over HAND; chance uses card IDs.

        Validate before the C++ mutation, including for off-menu actions.
        """
        if type(action) is not int or self.is_terminal():
            raise ValueError("Expected an integer action at a nonterminal state")
        chance = self.is_chance_node()
        if chance:
            valid = action in self._native.legal_actions()
        else:
            fold, call, minimum, maximum = self._bounds()
            valid = ((action == 0 and fold) or (action == 1 and call)
                     or (minimum is not None and minimum <= action <= maximum))
        if not valid:
            raise ValueError(f"Illegal native action {action}")
        self._native.apply_action(action)
        self._feature_cache.clear()
        code = 0.0 if chance else (-1.0 if action == 0 else
                                    -0.5 if action == 1 else action / STACK_CHIPS)
        self._public_history += (code,)
        if chance:
            self._cards += (action,)
        self._bounds_cache = self._spent_cache = self._menu_cache = None
        if not chance and self.is_chance_node():
            self._street_start = self._spent()

    def apply_street_raise_to(self, amount):
        """Exact cumulative contribution on this street (Slumbot-style units)."""
        if self.current_player() not in (0, 1) or type(amount) is not int or amount < 0:
            raise ValueError("Street raise requires an integer amount at a decision")
        native = self._street_start[self.current_player()] + amount
        if native < 2:
            raise ValueError("Raise amount cannot denote fold/check/call")
        self.apply_native_action(native)

    def decision_view(self, player=None):
        actor = self.current_player()
        player = actor if player is None else player
        if actor not in (0, 1) or player not in (0, 1):
            raise ValueError("Decision views require a player and a decision state")
        spent = self._spent()
        board = self._cards[4:]
        street = 0 if not board else len(board) - 2
        _, _, minimum, maximum = self._bounds()
        return DecisionView(
            player=player, actor=actor, street=street,
            hole_cards=self._cards[2 * player:2 * player + 2],
            board_cards=board, contributions=spent,
            street_contributions=tuple(x - y for x, y in zip(spent, self._street_start)),
            stacks=tuple(STACK_CHIPS - x for x in spent), pot=sum(spent),
            to_call=max(spent) - spent[actor], min_raise_to=minimum, max_raise_to=maximum,
            legal_mask=tuple(self.legal_actions_mask()), native_actions=self._menu(),
            public_history=self._public_history, action_menu=self._game.action_menu)

    def critic_hole_cards(self):
        """Training-only private state; never included in DecisionView."""
        return (self._cards[:2], self._cards[2:4])


def load_hunl_game(action_menu=DEFAULT_ACTION_MENU):
    return HUNLGame(action_menu)
