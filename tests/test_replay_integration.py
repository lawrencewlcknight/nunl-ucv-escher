"""Production capacities, actual encoder columns and overflow-safe metrics."""
import numpy as np
import torch
from hunl_escher.config import baseline_config, smoke_config, make_solver
from hunl_escher.train import learner_metrics
from test_hunl import state_at_street
from unbiased_escher.replay_grouping import retained_context_counts


def test_replay_capacity_and_collection_metrics():
    torch.set_num_threads(1)
    c=baseline_config()
    assert c['ave_policy_buffer_size'] == 8000000
    assert c['baseline_buffer_size'] == c['calibration_buffer_size'] == 1000000
    c=smoke_config(); c['ave_policy_buffer_size']=8
    solver=make_solver(c,seed=19)
    solver.iteration()
    metrics=learner_metrics(solver,1)
    buffer=solver.ave_policy_trainer.buffer
    assert buffer.cur_id > buffer.buffer_size
    assert metrics['policy_replay_rows'] == 8
    assert metrics['policy_replay']['seen'] == buffer.cur_id
    assert metrics['policy_replay']['not_retained'] == buffer.cur_id-8
    memory=metrics['allocated_learner_replay_bytes']
    assert memory['total'] == sum(v for k,v in memory.items() if k != 'total')
    for trainer in solver.regret_trainers:
        counts=trainer.last_fit_stats['replay']
        assert counts['retained'] == min(counts['seen'],counts['capacity'])
        assert np.sum(counts['retained_by_player_street']) == counts['retained']
    solver.train_average_policy()
    coverage=solver.ave_policy_trainer.last_replay_diagnostics
    assert coverage['retained'] == 8
    assert np.sum(coverage['retained_by_player_street']) == 8
    assert np.sum(coverage['unique_by_player_street']) == coverage['unique_groups']


def test_replay_columns_decode_real_states_for_both_players_and_all_streets():
    solver=make_solver(smoke_config())
    trainer=solver.ave_policy_trainer
    columns=solver.feature_encoder.replay_context_columns()
    for street in range(4):
        state=state_at_street(street)
        for player in range(2):
            feature=solver.feature_encoder.information_state(state,player)
            assert int(feature[columns[0]].argmax()) == player
            assert int(feature[columns[1]].argmax()) == street
            trainer.buffer.add(feature, np.ones(trainer.output_size)/trainer.output_size,
                               np.ones(trainer.output_size), 1)
    assert retained_context_counts(trainer.buffer,columns) == [[1]*4,[1]*4]
