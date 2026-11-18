"""Exact flop/turn all-in call values under a stipulated uniform opponent range."""
from itertools import combinations

import numpy as np

from ..strategic_features import rank_cards
from .engine import check_time, probabilities


def exact_all_in(state, policy, *, deadline=float('inf')):
    view = state.decision_view()
    if view.street not in (1, 2) or set(state.legal_actions()) != {0, 1}:
        raise ValueError('Requires a flop/turn all-in fold-or-call decision')
    remaining = [c for c in range(52) if c not in (*view.hole_cards, *view.board_cards)]
    missing = 5-len(view.board_cards)
    total_share, count, minimum, maximum, runouts = 0., 0, 1., 0., 0
    # Equivalent to first uniformly dealing a compatible opponent hand and
    # then all legal runouts. Each disjoint (hand, runout) has equal mass.
    for extra in combinations(remaining, missing):
        check_time(deadline)
        board = (*view.board_cards, *extra)
        hands = np.asarray(list(combinations([c for c in remaining if c not in extra], 2)))
        own = rank_cards([[*view.hole_cards, *board]])[0][0]
        ranks = rank_cards(np.column_stack((hands, np.tile(board, (len(hands), 1)))))[0]
        shares = (own > ranks) + .5*(own == ranks)
        total_share += float(shares.sum())
        count += len(shares)
        minimum, maximum = min(minimum,float(shares.min())), max(maximum,float(shares.max()))
        runouts += 1
    equity = total_share/count
    pot = (view.pot+view.to_call)/100
    spent = (view.contributions[view.player]+view.to_call)/100
    folded = -view.contributions[view.player]/100
    called = pot*equity-spent
    p = probabilities(policy, view)
    deployed = p[0]*folded+p[1]*called
    worst_advantage, best_advantage = pot*minimum-spent-folded, pot*maximum-spent-folded
    return dict(range='uniform compatible hands and runouts; not history-conditioned',
        exact_joint_outcomes=count, runouts=runouts, equity=equity,
        call_threshold=view.to_call/(view.pot+view.to_call),
        fold_value_bb=folded, call_value_bb=called, call_advantage_bb=called-folded,
        deployed_value_bb=float(deployed), local_gap_bb=float(max(folded,called)-deployed),
        guaranteed_cost_bb=float(p[0]*max(0,worst_advantage)+p[1]*max(0,-best_advantage)))
