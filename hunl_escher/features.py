"""Suit-canonical cards and an explicit, compact, exact-chip public history.

The policy sees only its own cards and public events. Critics additionally see
both players' hole cards. Flop, turn and river occupy separate channels; merging
all five public cards would discard information about earlier decisions.
"""
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .game import DEFAULT_ACTION_MENU, STACK_CHIPS, load_hunl_game
from .strategic_features import (CARD_NAMES, BOARD_CACHE_SIZE, HAND_CACHE_SIZE,
                                 card_features, betting_features, economics_names)
from vr_deep_cfr.solver import SonnetLinear, ZeroInitLinear

ENCODER_ID = "hunl_strategic_public_state_v3"


@dataclass(frozen=True)
class FeatureLayout:
    name: str
    total_size: int
    card_size: int

    @property
    def context_size(self):
        return self.total_size - self.card_size

    def to_dict(self):
        return dict(name=self.name, total_size=self.total_size, card_size=self.card_size)


def _canonicalise_suits(*channels):
    signatures = [tuple(float(x) for channel in channels for x in channel[suit])
                  for suit in range(4)]
    order = sorted(range(4), key=lambda suit: signatures[suit], reverse=True)
    return tuple(channel[order].reshape(-1) for channel in channels)


class HUNLFeatureEncoder:
    encoder_id = ENCODER_ID

    def __init__(self, action_menu=DEFAULT_ACTION_MENU):
        game = load_hunl_game(action_menu)
        self.action_menu = game.action_menu
        self.raw_size = game.information_state_tensor_size()
        self.history_slots = game.max_game_length()
        self.action_size = game.num_distinct_actions()
        # observer2, actor2, street4, stacks2, spent2, street-spent2, pot/call2,
        # raise bounds2, history length1, legal mask A, raise-to A, history H.
        context_size = 19 + 2 * self.action_size + self.history_slots
        self.policy_prefix_size = 208 + context_size
        self.critic_prefix_size = 260 + context_size
        extra = len(economics_names(self.action_menu))
        self.policy_layout = FeatureLayout("player_information_state", self.policy_prefix_size + len(CARD_NAMES) + extra, 208)
        self.full_state_layout = FeatureLayout("critic_full_state", self.critic_prefix_size + 2 * len(CARD_NAMES) + extra, 260)

    @property
    def policy_size(self):
        return self.policy_layout.total_size

    @property
    def full_state_size(self):
        return self.full_state_layout.total_size

    def metadata(self):
        return {"id": self.encoder_id, "version": 3,
                "raw_open_spiel_size": self.raw_size,
                "policy_layout": self.policy_layout.to_dict(),
                "full_state_layout": self.full_state_layout.to_dict(),
                "board_channels": ["flop", "turn", "river"],
                "bet_amount_scale_chips": STACK_CHIPS,
                "action_menu": list(self.action_menu),
                "history_slots": self.history_slots,
                "base_encoder_id": "hunl_exact_public_state_v2",
                "base_policy_prefix_size": self.policy_prefix_size,
                "base_full_state_prefix_size": self.critic_prefix_size,
                "strategic_card_features": list(CARD_NAMES),
                "strategic_economics_features": list(economics_names(self.action_menu)),
                "policy_extra_order": "own card features; actor economics with off-turn strength interactions zero",
                "critic_extra_order": "P0 card features; P1 card features; actor economics",
                "strength_prior": "uniform legal opponent holdings; CURRENT best-five rank, not runout equity",
                "draw_semantics": "visible-card pattern completion; not clean winning outs; zero on river",
                "caches_per_process": {"board_tables": BOARD_CACHE_SIZE, "card_vectors": HAND_CACHE_SIZE},
                "grouping": "unchanged exact information states; additive descriptors, no lossy buckets",
                "history_codes": {"chance": 0, "fold": -1, "check_call": -0.5,
                                  "raise": "hand_raise_to / stack"}}

    @staticmethod
    def _cards(cards):
        result = np.zeros((4, 13), dtype=np.float32)
        for card in cards:
            result[card % 4, card // 4] = 1
        return result

    def _board(self, view):
        return [self._cards(view.board_cards[:3]), self._cards(view.board_cards[3:4]),
                self._cards(view.board_cards[4:5])]

    def _view(self, state, player=None):
        if state.get_game().action_menu != self.action_menu:
            raise ValueError("State action menu differs from the encoder contract")
        return state.decision_view(player)

    def _context(self, view):
        if len(view.public_history) > self.history_slots:
            raise ValueError("Public history exceeds the pinned OpenSpiel bound")
        history = np.zeros(self.history_slots, dtype=np.float32)
        history[:len(view.public_history)] = view.public_history
        player = np.eye(2, dtype=np.float32)[view.player]
        street = np.eye(4, dtype=np.float32)[view.street]
        amounts = np.asarray((*view.stacks, *view.contributions,
                              *view.street_contributions, view.pot / 2,
                              view.to_call, view.min_raise_to or 0,
                              view.max_raise_to or 0), dtype=np.float32) / STACK_CHIPS
        sizes = np.asarray([a if a is not None and a >= 2 else 0
                            for a in view.native_actions], dtype=np.float32) / STACK_CHIPS
        return np.concatenate((player, np.eye(2, dtype=np.float32)[view.actor], street, amounts,
                               [len(view.public_history) / self.history_slots],
                               view.legal_mask, sizes, history))

    def information_state(self, state, player=None):
        return self.encode_decision(self._view(state, player))

    def encode_decision(self, view):
        """Identical input path for OpenSpiel training and remote public observations."""
        if view.action_menu != self.action_menu:
            raise ValueError("Observation action menu differs from the encoder contract")
        if view.player not in (0, 1) or view.actor not in (0, 1):
            raise ValueError("Invalid observation player")
        if len(view.hole_cards) != 2 or len(view.board_cards) not in (0, 3, 4, 5):
            raise ValueError("Invalid visible card counts")
        cards = (*view.hole_cards, *view.board_cards)
        if len(set(cards)) != len(cards) or any(type(c) is not int or not 0 <= c < 52 for c in cards):
            raise ValueError("Invalid or duplicate visible cards")
        if len(view.legal_mask) != self.action_size or not any(view.legal_mask):
            raise ValueError("Invalid decision mask")
        own = card_features(view.hole_cards, view.board_cards)
        return np.concatenate((*_canonicalise_suits(self._cards(view.hole_cards),
                                                    *self._board(view)),
                               self._context(view), own, betting_features(view, own))).astype(np.float32)

    def full_state(self, state):
        if state.is_terminal():
            return np.zeros(self.full_state_size, dtype=np.float32)
        view = self._view(state)
        private = state.critic_hole_cards()
        holes = [self._cards(cards) for cards in private]
        extras = [card_features(cards, view.board_cards) for cards in private]
        return np.concatenate((*_canonicalise_suits(*holes, *self._board(view)),
                               self._context(view), *extras,
                               betting_features(view, extras[view.actor]))).astype(np.float32)


def make_feature_encoder(encoder_id=ENCODER_ID, action_menu=DEFAULT_ACTION_MENU):
    if encoder_id != ENCODER_ID:
        raise ValueError(f"Unsupported HUNL encoder: {encoder_id}")
    return HUNLFeatureEncoder(action_menu)


def encoder_from_metadata(metadata):
    if metadata is None:
        raise ValueError("A HUNL policy must declare its encoder")
    encoder = make_feature_encoder(metadata.get("id"), metadata.get("action_menu", ()))
    if metadata != encoder.metadata():
        raise ValueError("Checkpoint encoder differs from the versioned HUNL contract")
    return encoder


class StructuredHUNLMLP(nn.Module):
    """Small CPU-friendly two-branch MLP for cards and betting context."""

    checkpoint_model_type = "structured_hunl_mlp_v1"

    def __init__(
        self,
        layout: FeatureLayout,
        hidden_layers,
        output_size: int,
        *,
        branch_width: int,
    ):
        super().__init__()
        hidden_layers = [int(width) for width in hidden_layers]
        if not hidden_layers or int(branch_width) <= 0:
            raise ValueError("Structured HUNL MLP requires positive hidden widths")
        self.layout = layout
        self.input_size = int(layout.total_size)
        self.output_size = int(output_size)
        self.hidden_layers = hidden_layers
        self.branch_width = int(branch_width)
        self.card_encoder = SonnetLinear(
            layout.card_size, self.branch_width, activation=True
        )
        self.context_encoder = SonnetLinear(
            layout.context_size, self.branch_width, activation=True
        )
        widths = [2 * self.branch_width, *hidden_layers]
        self.trunk = nn.Sequential(
            *[
                SonnetLinear(input_width, output_width, activation=True)
                for input_width, output_width in zip(widths[:-1], widths[1:])
            ]
        )
        self.output = ZeroInitLinear(hidden_layers[-1], self.output_size, False)

    def reset_parameters(self) -> None:
        self.card_encoder.reset_parameters()
        self.context_encoder.reset_parameters()
        for layer in self.trunk:
            layer.reset_parameters()
        self.output.reset_parameters()

    def checkpoint_metadata(self) -> dict[str, object]:
        return {
            "type": self.checkpoint_model_type,
            "layout": self.layout.to_dict(),
            "hidden_layers": list(self.hidden_layers),
            "branch_width": self.branch_width,
            "output_size": self.output_size,
        }

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        cards = inputs[..., : self.layout.card_size]
        context = inputs[..., self.layout.card_size :]
        joined = torch.cat(
            [self.card_encoder(cards), self.context_encoder(context)], dim=-1
        )
        return self.output(self.trunk(joined))
