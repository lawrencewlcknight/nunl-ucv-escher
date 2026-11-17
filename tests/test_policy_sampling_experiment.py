"""Unbiased loss/gradient scaling, legacy behavior, and frozen-artifact workflow."""
from copy import deepcopy
import random
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from unbiased_escher.policy_sampling import PolicyGroupSampler
from unbiased_escher.holdem_fitting import fit_validated_policy, policy_validation_mask
from hunl_escher.policy_replay_artifacts import load_policy_replay, read_metadata, policy_records
from hunl_escher.policy_sampling_experiment import run_fit
from hunl_escher.policy_sampling_evaluate import run_evaluation
from hunl_escher.checkpointing import save_policy_checkpoint, LoadedHUNLPolicy
from hunl_escher.config import baseline_config
from hunl_escher.features import make_feature_encoder, StructuredHUNLMLP
from hunl_escher.game import load_hunl_game, serialisable_game_definition
from hunl_escher.state_storage import save_sharded_training_state, STATE_TYPE


def test_expected_loss_and_gradient_match_with_validation_partition():
    torch.manual_seed(3)
    model = torch.nn.Linear(2, 3).double()
    x = torch.randn(5, 2, dtype=torch.float64)
    targets = torch.softmax(torch.randn(5, 3, dtype=torch.float64), -1)
    weights = torch.tensor([200., .2, 5., 3., 1.], dtype=torch.float64)
    ids = np.array([0, 2, 4])  # Noncontiguous training partition.
    gradients = []
    sampler = PolicyGroupSampler(ids, weights, 2, mode="objective_mass", seed=9)
    probabilities = torch.from_numpy(np.diff(np.r_[0., sampler.cdf]))
    for mass_sampling in (False, True):
        model.zero_grad()
        losses = -(targets[ids] * torch.log_softmax(model(x[ids]), -1)).sum(-1)
        loss = ((losses * weights[ids]).mean() if not mass_sampling else
                (probabilities * losses * sampler.factor).sum())
        loss.backward()
        gradients.append((loss.detach(), [p.grad.clone() for p in model.parameters()]))
    torch.testing.assert_close(gradients[0][0], gradients[1][0], atol=1e-12, rtol=1e-12)
    for a, b in zip(gradients[0][1], gradients[1][1]):
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)


def test_mass_draw_frequencies_and_no_double_weighting():
    weights = torch.tensor([1., 99., 1., 1.])
    ids = np.array([1, 2, 3])
    sampler = PolicyGroupSampler(ids, weights, 2, mode="objective_mass", seed=77)
    counts = np.zeros(4, dtype=int)
    for _ in range(10000):
        idx, factor = sampler.sample()
        np.add.at(counts, idx.numpy(), 1)
        assert factor.ndim == 0 and factor.item() == pytest.approx(101 / 3)
    np.testing.assert_allclose(counts[ids] / counts.sum(), [99 / 101, 1 / 101, 1 / 101], atol=.003)


def test_uniform_sampler_keeps_legacy_rng_and_full_batch_is_exact():
    ids, weights = np.arange(20)[::2], torch.arange(1, 21).float()
    random.seed(19)
    expected = [ids[random.sample(range(len(ids)), 4)] for _ in range(5)]
    expected_state = random.getstate()
    random.seed(19)
    sampler = PolicyGroupSampler(ids, weights, 4)
    for row in expected:
        idx, factors = sampler.sample()
        np.testing.assert_array_equal(idx, row)
        torch.testing.assert_close(factors, weights[row])
    assert random.getstate() == expected_state
    for mode in ("uniform", "objective_mass"):
        idx, factors = PolicyGroupSampler(ids, weights, 20, mode=mode).sample()
        np.testing.assert_array_equal(idx, ids)
        torch.testing.assert_close(factors, weights[ids])


def test_default_fit_matches_legacy_updates_and_selection():
    torch.manual_seed(8)
    x = torch.randn(80, 4)
    targets = torch.softmax(torch.randn(80, 3), -1)
    masks, weights = torch.ones(80, 3), torch.arange(1, 81).float() / 10
    data = (x, targets, masks, weights)
    initial = torch.nn.Linear(4, 3)
    train = np.flatnonzero(~policy_validation_mask(x, .05))
    model = deepcopy(initial)
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    random.seed(27)
    for _ in range(7):
        idx = torch.as_tensor(train[random.sample(range(len(train)), 8)])
        loss = (-(targets[idx] * torch.log_softmax(model(x[idx]), -1)).sum(-1) * weights[idx]).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    expected_rng = random.getstate()
    candidate = deepcopy(initial)
    trainer = SimpleNamespace(model=candidate, optimizer=torch.optim.Adam(candidate.parameters(), lr=.003),
        _grouped_training_data=lambda _: data, validation_fraction=.05, validation_steps=[7],
        train_steps=7, batch_size=8)
    random.seed(27)
    fit_validated_policy(trainer, 10)
    for a, b in zip(model.parameters(), candidate.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for key, values in optimizer.state_dict()["state"].items():
        for name, value in values.items():
            torch.testing.assert_close(value, trainer.optimizer.state_dict()["state"][key][name], rtol=0, atol=0)
    assert random.getstate() == expected_rng


@pytest.fixture
def frozen_source(tmp_path):
    torch.set_num_threads(1)
    config = baseline_config()
    config.update(average_policy_network_layers=[8, 8], average_policy_branch_width=4,
                  ave_policy_batch_size=32, average_policy_train_steps=8,
                  policy_validation_steps=[4, 8])
    encoder = make_feature_encoder(config["feature_encoder_id"], config["action_menu"])
    game = load_hunl_game(config["action_menu"])
    actions = game.num_distinct_actions()
    model = StructuredHUNLMLP(encoder.policy_layout, [8, 8], actions, branch_width=4)
    rng = np.random.default_rng(22)
    n = 400
    features = rng.normal(size=(n, encoder.policy_layout.total_size)).astype(np.float32)
    player, street = encoder.replay_context_columns()
    features[:, player] = np.eye(2)[np.arange(n) % 2]
    features[:, street] = np.eye(4)[np.arange(n) % 4]
    # A repeated group plus hundreds of singleton groups, with legal soft targets.
    features[:100] = features[0]
    targets = rng.dirichlet(np.arange(1, actions + 1), n).astype(np.float32)
    solver = SimpleNamespace(game=game, ave_policy_trainer=SimpleNamespace(model=model),
        feature_encoder=encoder, num_iteration=10, episode=200, nodes_touched=1000,
        infostate_size=encoder.policy_layout.total_size, action_size=actions,
        average_policy_network_layers=[8, 8], network_layers=[8, 8])
    policy = save_policy_checkpoint(solver, tmp_path / "source.pkl", seed=1, config=config, checkpoint_row={})
    metadata = dict(type=STATE_TYPE, schema_version=1, config=config,
        repository_commit="synthetic-test", game=serialisable_game_definition(), encoder=encoder.metadata(),
        solver=dict(num_iteration=10), average_policy_trainer=dict(model=model.state_dict(),
        buffer=dict(cur_id=900, size=n, infostate=features, q_value=targets,
                    q_value_mask=np.ones((n, actions), dtype=np.float32),
                    iteration=rng.integers(1, 11, (n, 1)).astype(np.float32))),
        unused_critic=np.zeros((5, 10), dtype=np.float32))
    state = tmp_path / "state"
    save_sharded_training_state(state, metadata)
    manifest, meta = read_metadata(state)
    selected = {r["path"] for r in policy_records(manifest, meta).values()}
    # Selective policy loading must not require unrelated critic replay.
    for r in manifest["array_shards"]:
        if r["path"] not in selected:
            (state / r["path"]).unlink()
    return state, policy


def test_selective_replay_checksum_and_end_to_end_refit(frozen_source, tmp_path):
    state, policy = frozen_source
    buffer, _, _ = load_policy_replay(state)
    assert isinstance(buffer.infostate_buf, np.memmap) and not buffer.infostate_buf.flags.writeable
    result = run_fit(state, policy, tmp_path / "fit", seeds=[17, 18], threads=1, smoke=True)
    assert result["status"] == "COMPLETE" and len(result["results"]) == 4
    for seed in (17, 18):
        rows = [r for r in result["results"] if r["fit_seed"] == seed]
        assert rows[0]["initial_model_sha256"] == rows[1]["initial_model_sha256"]
        for row in rows:
            loaded = LoadedHUNLPolicy(load_hunl_game(), tmp_path / "fit" / row["checkpoint"])
            assert loaded.checkpoint["policy_refit"]["arm"] == row["arm"]
            assert np.isfinite(row["validation"]["overall"]["weighted_kl"])
    evaluation = run_evaluation(tmp_path / "fit", tmp_path / "evaluation", workers=1, pilot_pairs=4, minimum=4, maximum=4, max_seconds=60)
    assert evaluation["status"] == "COMPLETE"
    assert evaluation["smoke"] and not evaluation["result"]["confirmatory"]
    record = next(iter(policy_records(*read_metadata(state)).values()))
    with (state / record["path"]).open("r+b") as handle:
        handle.seek(-1, 2)
        handle.write(b"x")
    with pytest.raises(ValueError, match="checksum"):
        load_policy_replay(state)


def test_prepare_fetches_only_policy_shards_and_reuses_verified_files(frozen_source, tmp_path, monkeypatch):
    from pathlib import Path
    from hunl_escher import policy_replay_artifacts as artifacts
    state, _ = frozen_source
    calls = []

    def transfer(command, **kwargs):
        relative = command[3].removeprefix("gs://test/checkpoint/")
        calls.append(relative)
        shutil.copyfile(state / relative, Path(command[4]))

    monkeypatch.setattr(artifacts, "run_transfer", transfer)
    target = tmp_path / "subset"
    plan = artifacts.prepare("gs://test/checkpoint", target, metadata_only=True)
    assert calls == ["manifest.json", "metadata.pt"]
    assert plan["partial_checkpoint"]
    calls.clear()
    artifacts.prepare("gs://test/checkpoint", target)
    assert set(calls) == {"manifest.json", *plan["files"]}
    buffer, _, _ = artifacts.load_policy_replay(target)
    assert len(buffer.infostate_buf) == 400
    calls.clear()
    artifacts.prepare("gs://test/checkpoint", target)
    assert calls == ["manifest.json"]


@pytest.fixture
def refit_experiment(frozen_source, tmp_path):
    state, policy = frozen_source
    root, evaluation = tmp_path / "refit", tmp_path / "local"
    run_fit(state, policy, root, seeds=[17, 18], threads=1, smoke=True)
    run_evaluation(root, evaluation, workers=1, pilot_pairs=4, minimum=4, maximum=4, max_seconds=60)
    return root, evaluation


def test_slumbot_fresh_arms_pilot_exclusion_and_predeclared_seed(refit_experiment, tmp_path):
    from hunl_escher.policy_sampling_slumbot import run_benchmark
    from hunl_escher.api import SlumbotClient
    from test_remote import MockServer
    root, evaluation = refit_experiment
    server = MockServer('slumbot')
    result = run_benchmark(root, evaluation, tmp_path / "slumbot", smoke=True,
        client_factory=lambda _, menu: SlumbotClient(action_menu=menu, transport=server))
    assert result['status'] == 'COMPLETE'
    assert result['fit_seed'] == 17
    assert result['completed_hands_per_arm'] == {'uniform': 8, 'objective_mass': 8}
    assert result['improvement']['observations'] == [8, 8]
    assert not result['improvement']['confirmatory']
    assert server.number == 24  # Four pilot and eight confirmation hands, independently for each arm.
    assert 'SECRET_TOKEN' not in (tmp_path / 'slumbot/hands.jsonl').read_text()


def test_slumbot_gate_prevents_api_calls_and_live_smoke(refit_experiment, tmp_path):
    from hunl_escher.policy_sampling_slumbot import run_benchmark
    root, evaluation = refit_experiment
    calls = []
    with pytest.raises(ValueError, match='local primary'):
        run_benchmark(root, evaluation, tmp_path / 'blocked', client_factory=lambda *a: calls.append(a))
    assert not calls and not (tmp_path / 'blocked').exists()
    with pytest.raises(ValueError, match='offline'):
        run_benchmark(root, evaluation, tmp_path / 'live-smoke', smoke=True)


def test_slumbot_ambiguous_post_is_not_retried_or_scored(refit_experiment, tmp_path):
    from hunl_escher.policy_sampling_slumbot import run_benchmark
    from hunl_escher.api import SlumbotClient, AmbiguousOutcome
    from test_remote import MockServer
    root, evaluation = refit_experiment

    class Broken(MockServer):
        failures = 0
        def request(self, method, path, body=None):
            if self.number == 11 and path == '/slumbot/api/new_hand':
                self.failures += 1
                raise AmbiguousOutcome('Uncertain outcome')
            return super().request(method, path, body)

    server = Broken('slumbot')
    result = run_benchmark(root, evaluation, tmp_path / 'partial', smoke=True,
        client_factory=lambda _, menu: SlumbotClient(action_menu=menu, transport=server))
    assert result['status'] == 'INCOMPLETE_PROVIDER_ERROR'
    assert server.failures == 1
    assert sum(result['completed_hands_per_arm'].values()) == 3
    assert result['improvement'] is None  # One arm has only one completed main hand.


def test_refit_checksum_and_local_report_identity_are_required(refit_experiment, tmp_path):
    import json
    from hunl_escher.policy_sampling_slumbot import verified_gate
    root, evaluation = refit_experiment
    manifest = json.loads((evaluation / 'manifest.json').read_text())
    manifest['experiment_sha256'] = '0' * 64
    (evaluation / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='does not match'):
        verified_gate(root, evaluation, smoke=True)


def test_shared_deals_are_aggregated_within_group(monkeypatch):
    from hunl_escher import policy_sampling_evaluate as evaluator
    monkeypatch.setattr(evaluator, '_error', None)
    monkeypatch.setattr(evaluator, '_game', object())
    monkeypatch.setattr(evaluator, '_agents', {'objective_mass_1': 1, 'uniform_1': 2,
                                            'objective_mass_2': 3, 'uniform_2': 4})
    monkeypatch.setattr(evaluator, 'duplicate_pair', lambda game, agents, seed, pair: [agents[0], agents[1]])
    rows = evaluator.batch((123, 0, 2, [1, 2]))
    assert len(rows) == 2 and all(r['mean_bb'] == 2.5 for r in rows)
