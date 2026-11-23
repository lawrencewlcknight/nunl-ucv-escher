"""Pinned-policy benchmarks: real offline wire adapters and strict hand accounting."""
import json
import shutil
from types import SimpleNamespace

import pytest

from hunl_escher import p128_benchmark as benchmark
from hunl_escher.api import AmbiguousOutcome, GTOWizardClient, SlumbotClient
from hunl_escher.checkpointing import sha256_file
from hunl_escher.remote import ProtocolError
from test_policy_sampling_experiment import frozen_source
from test_remote import MockServer
from test_gtow_p32 import Server


@pytest.fixture
def policy(frozen_source, tmp_path):
    _, source = frozen_source
    root = tmp_path / 'p128'
    root.mkdir()
    shutil.copyfile(source, root / 'P128.pkl')
    marker = dict(label='P128', sha256=sha256_file(source), outer_iteration=10, active_seconds=100)
    (root / 'P128.json').write_text(json.dumps(marker))
    (root / 'continuation.json').write_text(json.dumps(dict(status='TRAINING_COMPLETE',
        final_recovery_verified=True, smoke=True, checkpoints={'P128': marker})))
    return root


def client_factory(provider, server):
    return lambda name, menu: (GTOWizardClient if provider == 'gtow' else SlumbotClient)(
        action_menu=menu, transport=server)


@pytest.mark.parametrize('provider', ['slumbot', 'gtow'])
def test_complete_offline_protocol_without_pilots(policy, tmp_path, provider):
    server = Server(pending=True) if provider == 'gtow' else MockServer('slumbot')
    report = benchmark.run_benchmark(policy, tmp_path / provider, provider=provider,
        smoke=True, client_factory=client_factory(provider, server))
    assert report['status'] == 'COMPLETE'
    assert report['completed_confirmation_hands'] == 8
    assert report['completed_pilot_hands'] == 0
    assert report['fresh_hand_attempts'] == 8
    assert not (tmp_path / provider / 'pilot_precision.json').exists()
    assert report['estimate']['observations'] == [8]
    assert report['estimate']['full_fixed_sample']
    assert report['opponent_version_verified'] == (provider == 'gtow')
    assert not report['unresolved_hand_attempt']
    events = [json.loads(line) for line in (tmp_path / provider / 'hands.jsonl').read_text().splitlines()]
    completed = [e for e in events if e['event'] == 'hand_complete' and e['phase'] == 'main']
    assert len(completed) == 8
    if provider == 'gtow':
        assert report['scores']['main'] == [e['adjusted_scores']['aivat_score_chips'] / 100 for e in completed]
        assert report['excluded_existing_hand_ids'] == [1]
    assert report['raw']['main'] == [e['winnings_bb'] for e in completed]


def test_slumbot_timeout_preserves_partial_sample_without_replacement(policy, tmp_path):
    class FailAfterSix(MockServer):
        def request(self, method, path, body=None):
            if path.endswith('/new_hand') and self.number == 6:
                self.requests.append((method, path, body))
                raise AmbiguousOutcome('simulated response timeout; mutation not retried')
            return super().request(method, path, body)
    server = FailAfterSix('slumbot')
    report = benchmark.run_benchmark(policy, tmp_path / 'partial', provider='slumbot',
        smoke=True, client_factory=client_factory('slumbot', server))
    assert report['status'] == 'INCOMPLETE_PROVIDER_ERROR'
    assert report['completed_pilot_hands'] == 0 and report['completed_confirmation_hands'] == 6
    assert report['fresh_hand_attempts'] == 7 and report['unresolved_hand_attempt']
    assert len([r for r in server.requests if r[1].endswith('/new_hand')]) == 7
    assert report['estimate']['observations'] == [6]
    assert not report['estimate']['full_fixed_sample']
    assert report['incomplete_sample_warning']
    assert json.loads((tmp_path / 'partial/report.json').read_text()) == report


def test_gto_quota_refusal_stops_without_retries_or_zero_scores(policy, tmp_path):
    server = Server(ambiguous=True)
    result = benchmark.run_benchmark(policy, tmp_path / 'quota', provider='gtow',
        smoke=True, client_factory=client_factory('gtow', server))
    assert result['fresh_hand_attempts'] == 1
    assert result['completed_confirmation_hands'] == 0 and result['estimate'] is None
    assert len([r for r in server.requests if r[0] == 'POST']) == 1


def test_opponent_version_change_invalidates_result(policy, tmp_path):
    report = benchmark.run_benchmark(policy, tmp_path / 'changed', provider='gtow', smoke=True,
        client_factory=client_factory('gtow', Server(version_change=True)))
    assert report['status'] == 'INVALID_OPPONENT_VERSION_CHANGE'
    assert not report['opponent_version_verified']
    assert not report['estimate']['full_fixed_sample']


def test_provenance_failure_precedes_any_api_call(policy, tmp_path):
    (policy / 'P128.pkl').write_bytes(b'corrupt')
    server = Server()
    with pytest.raises(ValueError, match='Policy hash'):
        benchmark.run_benchmark(policy, tmp_path / 'bad', provider='gtow', smoke=True,
            client_factory=client_factory('gtow', server))
    assert not server.requests


def test_production_rejects_synthetic_and_smoke_requires_offline_client(policy, tmp_path):
    with pytest.raises(ValueError, match='smoke'):
        benchmark.run_benchmark(policy, tmp_path / 'bad', provider='slumbot')
    with pytest.raises(ValueError, match='injected offline'):
        benchmark.run_benchmark(policy, tmp_path / 'bad', provider='gtow', smoke=True)


def test_slumbot_transport_uses_long_timeout_no_post_retry():
    client = benchmark.benchmark_client('slumbot', ('fold', 'call'))
    assert client.transport.timeout == 180 and client.transport.min_interval == .5
    assert client.transport.post_attempts == 1
    assert client.transport.retryable_post_statuses == ()


@pytest.mark.parametrize('provider,hand_count', [('gtow', 8000), ('slumbot', 20000)])
def test_full_production_budget_has_no_pilot(tmp_path, monkeypatch, provider, hand_count):
    # Exercise production loop and budget with cheap offline terminal hands.
    class Client:
        transport = SimpleNamespace(request=lambda *a, **k: dict(gto_wizard_version=2, total_hands=0))
        def __init__(self):
            self.calls = 0
        def _in_progress(self):
            return []
        def new_hand(self):
            self.calls += 1
            self.current = SimpleNamespace(hand_id=self.calls)
    client = Client()
    client.provider = provider
    monkeypatch.setattr(benchmark, 'verified_candidate', lambda *a, **k: ({}, {'policy_sha256': 'test'}))
    monkeypatch.setattr(benchmark, 'Agent', lambda **k: SimpleNamespace(checkpoint_hash='test', action_menu=[]))
    def complete(client, *args):
        client.new_hand()
        return .5, .5
    monkeypatch.setattr(benchmark, 'collect_hand', complete)
    report = benchmark.run_benchmark(tmp_path, tmp_path / 'full', provider=provider,
        client_factory=lambda *args: client)
    assert report['status'] == 'COMPLETE' and client.calls == hand_count
    assert report['completed_pilot_hands'] == 0 and report['completed_confirmation_hands'] == hand_count
    guard = benchmark.HandBudget(client, 1)
    guard.new_hand()
    with pytest.raises(ProtocolError, match='budget exhausted'):
        guard.new_hand()
    assert guard.attempted == 1


def test_deadline_stops_within_a_hand(policy, tmp_path, monkeypatch):
    clock = [0.]
    monkeypatch.setattr(benchmark.time, 'monotonic', lambda: clock[0])
    class SlowStart(MockServer):
        def request(self, method, path, body=None):
            result = super().request(method, path, body)
            if path.endswith('/new_hand'):
                clock[0] = 2.
            return result
    server = SlowStart('slumbot')
    result = benchmark.run_benchmark(policy, tmp_path / 'timeout', provider='slumbot', smoke=True,
        max_seconds=1, client_factory=client_factory('slumbot', server))
    assert result['status'] == 'INCOMPLETE_TIME_LIMIT'
    assert result['fresh_hand_attempts'] == 1 and result['unresolved_hand_attempt']
    assert result['completed_pilot_hands'] == 0
    assert not any(r[1].endswith('/act') for r in server.requests)
