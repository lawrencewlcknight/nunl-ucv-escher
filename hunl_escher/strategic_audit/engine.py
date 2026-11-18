"""Exact river integration against explicit, information-safe stress opponents.

Only the root action is optimized. All later hero decisions retain the candidate
policy. The hidden hand is integrated BEFORE root maximization. Opponents use
fold/call/jam, keeping the entire continuation finite and inexpensive even when
the candidate has the full seven-action menu. No branch is depth-truncated.
"""
from functools import lru_cache
from itertools import combinations
import time

import numpy as np

from ..strategic_features import rank_cards

PANELS = ("calling_station", "value_pressure", "polarized_pressure")


class BudgetExpired(RuntimeError):
    pass


def check_time(deadline):
    if time.monotonic() >= deadline:
        raise BudgetExpired("Audit time budget exhausted")


def probabilities(policy, view):
    raw = policy.action_probabilities_from_view(view)
    if not isinstance(raw, dict) or any(type(k) is not int or not 0 <= k < len(view.legal_mask) for k in raw):
        raise ValueError("Policy returned invalid action IDs")
    p = np.asarray([raw.get(i, 0.) for i in range(len(view.legal_mask))], dtype=float)
    if (not np.isfinite(p).all() or (p < 0).any() or
            np.any(p[np.logical_not(view.legal_mask)] != 0) or abs(p.sum()-1) > 1e-6):
        raise ValueError("Policy must return finite, normalized legal probabilities")
    return p / p.sum()  # Only floating-point normalization, no smoothing.


class RulePolicy:
    def __init__(self, kind):
        if kind not in ("check_call", "random", "all_in", "fold"):
            raise ValueError("Unknown diagnostic rule policy")
        self.kind = kind

    def action_probabilities_from_view(self, view):
        legal = [i for i, x in enumerate(view.legal_mask) if x]
        if self.kind == "random":
            return {i: 1/len(legal) for i in legal}
        a = 1
        if self.kind == "all_in":
            jam = view.action_menu.index("all_in")
            a = jam if jam in legal else 1
        elif self.kind == "fold" and 0 in legal:
            a = 0
        return {i: float(i == a) for i in legal}


@lru_cache(maxsize=16)
def board_table(board):
    hands = np.array(list(combinations([c for c in range(52) if c not in board], 2)))
    ranks, _, _ = rank_cards(np.column_stack((hands, np.tile(board, (len(hands), 1)))))
    # Opponent's own uniform-random showdown equity. Hero's actual cards are
    # deliberately absent: this is a property of that opponent information set.
    strengths = np.empty(len(hands))
    for i, h in enumerate(hands):
        valid = ~np.isin(hands, h).any(axis=1)
        strengths[i] = np.mean((ranks[i] > ranks[valid]) + .5*(ranks[i] == ranks[valid]))
    for a in (hands, ranks, strengths):
        a.flags.writeable = False
    return hands, ranks, strengths


def river_distribution(view, panel):
    if view.street != 3 or panel not in PANELS:
        raise ValueError("Exact values require a river and a known panel")
    hands, ranks, strengths = board_table(tuple(view.board_cards))
    valid = ~np.isin(hands, view.hole_cards).any(axis=1)
    hands, ranks, strengths = hands[valid], ranks[valid], strengths[valid]
    hero_rank = rank_cards([[*view.hole_cards, *view.board_cards]])[0][0]
    weights = np.ones(len(hands))
    if panel == "value_pressure":
        weights = np.exp(4*(strengths-.5))
    elif panel == "polarized_pressure":
        weights = .1 + 8*(strengths-.5)**2
    return hands, weights/weights.sum(), np.sign(hero_rank-ranks), strengths


def opponent_probabilities(view, strengths, panel):
    """Frozen synthetic policy, using only own strength and public prices."""
    result = np.zeros((len(strengths), len(view.legal_mask)))
    jam = view.action_menu.index("all_in")
    if panel == "calling_station":
        result[:, 1] = 1
        return result
    odds = view.to_call / (view.pot + view.to_call)
    if panel == "value_pressure":
        aggressive = np.where(strengths >= .85, .9, np.where(strengths < .2, .05, .05))
    else:
        aggressive = np.where(strengths >= .8, .85, np.where(strengths < .25, .35, .02))
    if not view.legal_mask[jam]:
        aggressive = np.zeros(len(strengths))
    result[:, jam] = aggressive
    if view.to_call:
        # All probabilities are strictly defined, even off the panel's own reach.
        call = np.where(strengths >= odds, .9, .1)
        result[:, 1] = (1-aggressive)*call
        result[:, 0] = (1-aggressive)*(1-call)
    else:
        result[:, 1] = 1-aggressive
    return result


def exact_river(state, policy, panel, *, deadline=float("inf"), max_nodes=10000):
    view = state.decision_view()
    hands, prior, outcome, strengths = river_distribution(view, panel)
    hero = view.player
    nodes = 0

    def value(node):
        nonlocal nodes
        check_time(deadline)
        nodes += 1
        if nodes > max_nodes:
            raise RuntimeError("River tree exceeded node safety bound; no approximate score published")
        if node.is_terminal():
            if node.native_history()[-1].action == 0:
                return np.full(len(hands), node.returns()[hero]/100)
            spent = node._spent()
            return ((sum(spent)*((outcome > 0) + .5*(outcome == 0))) - spent[hero])/100
        if node.is_chance_node():
            raise ValueError("River audit unexpectedly reached chance")
        current = node.decision_view()
        if current.actor == hero:
            p = probabilities(policy, current)
            return sum((p[a]*value(node.child(a)) for a in node.legal_actions() if p[a] > 0),
                       np.zeros(len(hands)))
        # Never query a candidate on placeholder opponent cards.
        p = opponent_probabilities(current, strengths, panel)
        return sum((p[:, a]*value(node.child(a)) for a in node.legal_actions()
                    if np.any(p[:, a] > 0)), np.zeros(len(hands)))

    p = probabilities(policy, view)
    vectors = {a: value(state.child(a)) for a in state.legal_actions()}
    q = {a: float(prior @ v) for a, v in vectors.items()}
    deployed = sum(p[a]*q[a] for a in q)
    best = max(q.values())
    result = dict(panel=panel, compatible_hands=len(hands), nodes=nodes,
        action_values_bb={view.action_menu[a]: v for a, v in q.items()},
        deployed_value_bb=float(deployed), best_root_value_bb=best,
        local_gap_bb=max(0., float(best-deployed)),
        gap_pot_fraction=max(0., float(best-deployed))/(view.pot/100),
        action_costs_bb={view.action_menu[a]: max(0., best-v) for a, v in q.items()},
        showdown_equity=float(prior @ ((outcome > 0)+.5*(outcome == 0))),
        universal_fold_error_mass=0., universal_call_error_mass=0., guaranteed_cost_bb=0.)
    if view.to_call and state.child(1).is_terminal():
        advantage = vectors[1]-vectors[0]
        result.update(call_threshold=view.to_call/(view.pot+view.to_call),
                      range_call_advantage_bb=q[1]-q[0])
        if np.min(advantage) > 1e-10:
            result.update(universal_fold_error_mass=float(p[0]),
                          guaranteed_cost_bb=float(p[0]*np.min(advantage)))
        elif np.max(advantage) < -1e-10:
            result.update(universal_call_error_mass=float(p[1]),
                          guaranteed_cost_bb=float(p[1]*-np.max(advantage)))
    return result
