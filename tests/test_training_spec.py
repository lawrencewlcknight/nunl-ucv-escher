from copy import deepcopy
import numpy as np
import pytest
import torch
from hunl_escher.config import smoke_config, make_solver
from unbiased_escher.holdem_fitting import policy_validation_mask, weighted_policy_ce
from test_hunl import assert_models_equal

@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)

@pytest.mark.parametrize('rows', [5,19,270])
def test_cached_regret_fit_matches_uncached_and_rebuilds(rows):
    results = []
    for cached in (False,True):
        c=smoke_config(); c['cache_frozen_regret_targets']=cached
        solver=make_solver(c,seed=33)
        trainer=solver.regret_trainers[0]
        rng=np.random.default_rng(21)
        with torch.no_grad():
            for p in trainer.model.parameters(): p.uniform_(-.03,.03)
            trainer.target_model.load_state_dict(trainer.model.state_dict())
        for i in range(rows):
            trainer.buffer.add(rng.random(trainer.input_size,dtype=np.float32),
                rng.normal(size=trainer.output_size).astype(np.float32), [1] * (trainer.output_size-1) + [i%2], 2)
        trainer.train_model(2)
        first=deepcopy(trainer.model.state_dict())
        # New labels and a new target/iteration must invalidate the previous cache.
        trainer.buffer.q_value_buf[:min(rows,256)] *= -2
        trainer.train_model(3)
        results.append((first,deepcopy(trainer.model.state_dict()),
                        deepcopy(trainer.optimizer.state_dict()),deepcopy(trainer.buffer.rng.bit_generator.state)))
    assert_models_equal(results[0][:2],results[1][:2],exact=False)
    assert results[0][3] == results[1][3]
    for key, state in results[0][2]['state'].items():
        for field, value in state.items():
            torch.testing.assert_close(value,results[1][2]['state'][key][field],atol=1e-6,rtol=1e-5)


def test_policy_split_is_stable_and_groups_duplicates():
    trainer=make_solver(smoke_config()).ave_policy_trainer
    rng=np.random.default_rng(8)
    for i in range(100):
        feature=rng.random(trainer.input_size,dtype=np.float32)
        for iteration in (1,2): trainer.buffer.add(feature,np.full(trainer.output_size, 1/trainer.output_size),np.ones(trainer.output_size),iteration)
    features,targets,masks,weights=trainer._grouped_training_data(2)
    assert len(features)==100
    split=policy_validation_mask(features,.2)
    assert 0<sum(split)<100
    perm=rng.permutation(100)
    np.testing.assert_array_equal(policy_validation_mask(features[perm],.2),split[perm])
    # Later fits retain the same membership for existing information sets.
    more=torch.cat([features,torch.ones(1,trainer.input_size)])
    np.testing.assert_array_equal(policy_validation_mask(more,.2)[:100],split)


def test_policy_keeps_best_validation_weights_and_optimizer(monkeypatch):
    import unbiased_escher.holdem_fitting as fitting
    trainer=make_solver(smoke_config()).ave_policy_trainer
    rng=np.random.default_rng(10)
    for _ in range(100):
        trainer.buffer.add(rng.random(trainer.input_size,dtype=np.float32),np.full(trainer.output_size, 1/trainer.output_size),np.ones(trainer.output_size),1)
    trainer.validation_fraction=.2
    snapshots=[]
    def score(model,*args):
        snapshots.append((deepcopy(model.state_dict()),deepcopy(trainer.optimizer.state_dict())))
        return len(snapshots) # First fit is deliberately the best.
    monkeypatch.setattr(fitting,'weighted_policy_ce',score)
    trainer.train_model(1)
    assert trainer.last_fit_diagnostics['selected_update']==1
    assert trainer.last_fit_diagnostics['updates_completed']==2
    assert_models_equal([trainer.model.state_dict()],[snapshots[0][0]])
    for state in trainer.optimizer.state_dict()['state'].values(): assert state['step']==1


def test_validation_ce_uses_iteration_masses():
    model=torch.nn.Identity()
    features=torch.tensor([[2.,0.],[0.,1.],[0.,0.]])
    targets=torch.tensor([[0.,1.],[1.,0.],[.5,.5]])
    weights=torch.tensor([1.,9.,2.]); masks=torch.ones_like(features)
    actual=weighted_policy_ce(model,features,targets,masks,weights,np.arange(3),2)
    expected=(-(targets*torch.log_softmax(features,-1)).sum(-1)*weights).sum()/weights.sum()
    assert actual==pytest.approx(expected.item())




@pytest.mark.parametrize('values,legal,expected', [
    ([0,0,0], [0,1,2], [1/3,1/3,1/3]),
    ([-1,-1,-2], [0,1,2], [.5,.5,0]),
    ([-3,-1,-2], [0,1,2], [0,1,0]),
    ([10,-1,-1], [1,2], [0,.5,.5]),
    ([1,2,-1], [0,1,2], [1/3,2/3,0]),
])
def test_uniform_best_ties_only(values, legal, expected):
    trainer = make_solver(smoke_config()).regret_trainers[0]
    np.testing.assert_allclose(trainer.regret_matching(np.array(values,dtype=float),legal),expected)
