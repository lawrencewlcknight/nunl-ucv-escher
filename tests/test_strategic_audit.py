"""Analytical poker checks and independent native settlement/branch oracle."""
from dataclasses import replace
from itertools import combinations
import json
import time

import numpy as np
import pytest

from hunl_escher.remote import cards
from hunl_escher.strategic_audit.library import (
    make_library, validate_library, state_for, write_new, digest, board_class,
)
from hunl_escher.strategic_audit.engine import (
    RulePolicy, exact_river, river_distribution, probabilities, opponent_probabilities,
    BudgetExpired, PANELS,
)
from hunl_escher.strategic_audit.workflow import run, observation_checks


def position(hole='KsQs', board='AsTs4s9d2c', history='b300c/kk/kk/kb19700', hero=1):
    return dict(id='test', family='test', partition='development', hero=hero,
                hole=hole, board=board, history=history, tags=['test'])


def test_full_library_coverage_legality_and_frozen_generation():
    a, b = make_library(), make_library()
    assert digest(a) == digest(b)
    assert digest(a) != digest(make_library(seed=2))
    views = [state_for(p).decision_view() for p in a['positions']]
    assert {v.street for v in views} == {0, 1, 2, 3}
    assert {v.actor for v in views} == {0, 1}
    assert len(a['positions']) > 400
    assert any(v.to_call > v.pot-v.to_call for v in views)
    dev = {board_class(cards(p['board'])) for p in a['positions'] if p['partition']=='development' and p['board']}
    assess = {board_class(cards(p['board'])) for p in a['positions'] if p['partition']=='assessment'}
    assert not dev & assess
    duplicate = json.loads(json.dumps(a))
    duplicate['positions'].append(duplicate['positions'][0])
    with pytest.raises(ValueError, match='unique'):
        validate_library(duplicate)


@pytest.mark.parametrize('panel', PANELS)
def test_call_fold_equity_pot_odds_and_provable_nuts(panel):
    s = state_for(position())
    view = s.decision_view()
    result = exact_river(s, RulePolicy('fold'), panel)
    assert result['compatible_hands'] == 990
    assert result['showdown_equity'] == pytest.approx(1)
    assert result['universal_fold_error_mass'] == 1
    expected = ((view.pot+view.to_call)-view.to_call)/100
    assert result['range_call_advantage_bb'] == pytest.approx(expected)
    assert result['local_gap_bb'] == pytest.approx(expected)
    assert result['guaranteed_cost_bb'] == pytest.approx(expected)
    assert result['call_threshold'] == pytest.approx(view.to_call/(view.pot+view.to_call))
    assert exact_river(s, RulePolicy('check_call'), panel)['local_gap_bb'] == pytest.approx(0)


def test_board_royal_ties_and_sunk_costs():
    s = state_for(position(hole='2c3d', board='AsKsQsJsTs'))
    r = exact_river(s, RulePolicy('fold'), 'calling_station')
    assert r['showdown_equity'] == pytest.approx(.5)
    assert r['action_values_bb']['check_call'] == 0
    assert r['action_values_bb']['fold'] == pytest.approx(-3)
    assert r['guaranteed_cost_bb'] == 3


@pytest.mark.parametrize('history,hero', [('b300c/kk/kk/k',0), ('b300c/kk/kk/b300',0),
                                          ('b300c/kk/kk/kb600',1)])
def test_vector_tree_matches_independent_native_per_hand_oracle(history, hero):
    p = position(hole='KcQd', board='AsTs4s9d2c', history=history, hero=hero)
    s = state_for(p)
    candidate = RulePolicy('random')
    for panel in ('calling_station', 'polarized_pressure'):
        hands, prior, _, strengths = river_distribution(s.decision_view(), panel)
        # Native engine settles every leaf. A fixed stratified subset tests
        # arbitrary branch prices, seats, fold/call/raise and blocker legality.
        subset = np.arange(0, len(hands), 101)
        prior = prior[subset]/prior[subset].sum()
        expected = {a:0. for a in s.legal_actions()}
        for mass, index in zip(prior, subset):
            real = state_for(p, dummy=tuple(map(int, hands[index])))
            def scalar(node):
                if node.is_terminal():
                    return node.returns()[hero]/100
                view = node.decision_view()
                weights = (probabilities(candidate, view) if view.actor==hero else
                           opponent_probabilities(view, strengths[index:index+1], panel)[0])
                return sum(weights[a]*scalar(node.child(a)) for a in node.legal_actions() if weights[a]>0)
            for a in expected:
                expected[a] += mass*scalar(real.child(a))
        # Inject only the test distribution; native leaf settlement remains independent.
        from unittest.mock import patch
        original = river_distribution(s.decision_view(), panel)
        with patch('hunl_escher.strategic_audit.engine.river_distribution',
                   return_value=(hands[subset], prior, original[2][subset], strengths[subset])):
            actual = exact_river(s, candidate, panel)
        for a, ev in expected.items():
            assert actual['action_values_bb'][s.decision_view().action_menu[a]] == pytest.approx(ev, abs=1e-10)


def test_integrate_before_optimizing_no_clairvoyant_response():
    s = state_for(position(hole='AcQd', board='Ah7c6s4d3h'))
    r = exact_river(s, RulePolicy('random'), 'calling_station')
    view = s.decision_view()
    _, prior, outcomes, _ = river_distribution(view, 'calling_station')
    folded = -view.contributions[view.player]/100
    called = (view.pot+view.to_call)*((outcomes>0)+.5*(outcomes==0))/100-200
    assert r['best_root_value_bb'] == pytest.approx(max(folded, float(prior@called)))
    assert float(prior@np.maximum(folded, called)) > r['best_root_value_bb']


def test_privacy_order_and_suits():
    p = position()
    s = state_for(p)
    checks = observation_checks(p,s,RulePolicy('random'),suits=True)
    assert checks == dict(hidden_card_view_identical=True, hole_order_total_variation=0.,
                         flop_order_total_variation=0., max_suit_total_variation=0.)
    class RecordingPolicy(RulePolicy):
        def action_probabilities_from_view(self, view):
            assert view.hole_cards == s.decision_view().hole_cards
            assert view.actor == p['hero']
            return super().action_probabilities_from_view(view)
    exact_river(s, RecordingPolicy('random'), 'polarized_pressure')


def test_invalid_probabilities_and_nonriver_fail_closed():
    class Bad:
        def action_probabilities_from_view(self, view): return {1: float('nan')}
    s = state_for(position())
    with pytest.raises(ValueError, match='finite'):
        exact_river(s, Bad(), 'calling_station')
    with pytest.raises(ValueError, match='river'):
        exact_river(state_for(position(board='',hole='AsAh',history='',hero=0)), RulePolicy('random'),'calling_station')
    with pytest.raises(BudgetExpired):
        exact_river(s, RulePolicy('random'), 'calling_station', deadline=0)
    with pytest.raises(RuntimeError, match='safety'):
        exact_river(s, RulePolicy('random'), 'calling_station', max_nodes=1)


def test_workflow_resume_holdout_checksums_and_partial_reporting(tmp_path, monkeypatch):
    lib = tmp_path/'library.json'
    write_new(lib,make_library(smoke=True))
    specs = [('a','rule:random'), ('b','rule:random')]
    out = tmp_path/'audit'
    from hunl_escher.strategic_audit import workflow
    original = workflow.audit_position
    def timeout(*args,**kwargs): raise BudgetExpired('test limit')
    monkeypatch.setattr(workflow, 'audit_position',timeout)
    partial = run(lib,specs,out,max_seconds=30)
    assert partial['status']=='INCOMPLETE_TIME_LIMIT'
    assert partial['summary']['comparisons_suppressed']
    monkeypatch.setattr(workflow,'audit_position',original)
    complete = run(lib,specs,out,max_seconds=30,resume=True)
    assert complete['status']=='COMPLETE'
    assert all(x['gap_delta_b_minus_a_bb']==0 for x in complete['summary']['comparisons'])
    assert '<html' in (out/'report.html').read_text()
    with pytest.raises(FileExistsError): run(lib,specs,out)
    with pytest.raises(ValueError,match='identical'): run(lib,[('a','rule:fold')],out,resume=True)
    with pytest.raises(ValueError,match='unlock'): run(lib,specs,tmp_path/'assessment',partition='assessment')
    assessment=run(lib,specs,tmp_path/'assessment',partition='assessment',unlock_assessment=True,max_seconds=30)
    assert assessment['status']=='COMPLETE'
    record=next((out/'records').glob('*.json'))
    data=json.loads(record.read_text()); data['sha256']='wrong';record.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='checksum'):run(lib,specs,out,resume=True)
    assert json.loads((out/'report.json').read_text())['status']=='FAILED'


def test_real_checkpoint_roundtrip_and_information_checks(tmp_path):
    import torch
    from hunl_escher.config import smoke_config, make_solver
    from hunl_escher.checkpointing import save_policy_checkpoint, LoadedHUNLPolicy
    from hunl_escher.game import load_hunl_game
    from hunl_escher.strategic_audit.workflow import policy_spec, audit_position
    torch.set_num_threads(1)
    config = smoke_config()
    solver = make_solver(config)
    solver.iteration()
    solver.train_average_policy()
    checkpoint = tmp_path/'tiny.pkl'
    save_policy_checkpoint(solver, checkpoint, seed=0, config=config, checkpoint_row={})
    spec = policy_spec('tiny', str(checkpoint))
    assert spec['sha256'] and spec['iteration']==1
    policy = LoadedHUNLPolicy(load_hunl_game(),checkpoint)
    p = position(history='b300c/kk/kk/k',hero=0)
    record = audit_position(p,policy,tuple(spec['action_menu']),suits=True,deadline=time.monotonic()+60)
    assert len(record['river'])==3
    assert record['checks']['max_suit_total_variation'] < 1e-6
    assert record['checks']['hole_order_total_variation'] < 1e-6
    assert record['checks']['flop_order_total_variation'] < 1e-6
    s = state_for(p)
    available = [c for c in range(52) if c not in (*cards(p['hole']), *cards(p['board']))]
    other = state_for(p,dummy=tuple(available[-2:]))
    assert exact_river(s,policy,'value_pressure')==exact_river(other,policy,'value_pressure')
    lib = make_library(smoke=True)
    lib['positions'] = [p]
    path = tmp_path/'lib.json';write_new(path,lib)
    result = run(path,[('tiny',str(checkpoint))],tmp_path/'result',max_seconds=60)
    assert result['status']=='COMPLETE'


def test_opponent_policy_never_uses_actual_hero_cards():
    # Fix opponent's own private holding and public history; alter hero cards.
    first = state_for(position(history='b300c/kk/kk/k',hero=0),dummy=cards('8c8d'))
    second = state_for(position(hole='KcQd',history='b300c/kk/kk/k',hero=0),dummy=cards('8c8d'))
    for s in (first,second):s.apply_action(3)
    for panel in PANELS:
        a=opponent_probabilities(first.decision_view(),np.array([.4]),panel)
        b=opponent_probabilities(second.decision_view(),np.array([.4]),panel)
        np.testing.assert_array_equal(a,b)


def test_interruption_never_leaves_complete_report(tmp_path,monkeypatch):
    from hunl_escher.strategic_audit import workflow
    lib=tmp_path/'lib.json';write_new(lib,make_library(smoke=True))
    def interrupt(*a,**k):raise KeyboardInterrupt()
    monkeypatch.setattr(workflow,'audit_position',interrupt)
    out=tmp_path/'audit'
    with pytest.raises(KeyboardInterrupt):run(lib,[('a','rule:random')],out)
    report=json.loads((out/'report.json').read_text())
    assert report['status']=='FAILED' and report['summary']['comparisons_suppressed']
    assert not (out/'audit.lock').exists()


def test_turn_all_in_exact_runouts_and_native_settlement():
    from hunl_escher.strategic_audit.all_in import exact_all_in
    p=position(hole='AsKs',board='QsJsTs2h',history='b300c/kk/b19700',hero=0)
    s=state_for(p)
    r=exact_all_in(s,RulePolicy('fold'))
    assert r['runouts']==46 and r['exact_joint_outcomes']==45540
    assert r['equity']==1 and r['call_value_bb']==200
    assert r['local_gap_bb']==203 and r['guaranteed_cost_bb']==203
    # Independent native settlement for ALL legal (opponent, river) pairs.
    p=position(hole='AsQs',board='Js7s2d4c',history='b300c/kk/b19700',hero=0)
    result=exact_all_in(state_for(p),RulePolicy('check_call'))
    available=[c for c in range(52) if c not in (*cards(p['hole']),*cards(p['board']))]
    payoffs=[]
    for hand in combinations(available,2):
        actual=state_for(p,dummy=hand)
        actual.apply_action(1)
        for river in available:
            if river not in hand:
                final=actual.child(river)
                assert final.is_terminal()
                payoffs.append(final.returns()[p['hero']]/100)
    assert len(payoffs)==45540
    assert result['call_value_bb']==pytest.approx(np.mean(payoffs))
    assert 0 < result['equity'] < 1
    expected=(400*result['equity']-200)-(-3)
    assert result['call_advantage_bb']==pytest.approx(expected)
    assert result['call_threshold']==pytest.approx(19700/40000)
    with pytest.raises(BudgetExpired):exact_all_in(s,RulePolicy('fold'),deadline=0)
