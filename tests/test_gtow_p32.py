"""Exercise corrected-policy provenance and the full GTO wire protocol offline."""
import json
import pickle

import pytest

from hunl_escher.api import APIError, AmbiguousOutcome, GTOWizardClient
from hunl_escher.checkpointing import sha256_file
from hunl_escher.gtow_p32 import run_benchmark, verified_candidate
from hunl_escher.policy_sampling_experiment import run_fit
from test_policy_sampling_experiment import frozen_source
from test_remote import MockServer


@pytest.fixture
def refit(frozen_source, tmp_path):
    state, policy = frozen_source
    root = tmp_path / "refit"
    run_fit(state, policy, root, seeds=[1701], threads=1, smoke=True)
    return root


class Server(MockServer):
    def __init__(self, *, version_change=False, ambiguous=False, pending=False):
        super().__init__("gtow")
        self.version_change, self.ambiguous = version_change, ambiguous
        self.version_calls = 0
        if pending:
            self.request("POST", "/hands", {"game_name": "HUNL 200BB"})
        self.requests.clear()

    def request(self, method, path, body=None):
        if path.startswith("/results?"):
            self.requests.append((method, path, body))
            self.version_calls += 1
            return dict(gto_wizard_version=2 + int(self.version_change and self.version_calls > 1),
                        total_hands=self.number)
        if method == "GET" and self.state is None:
            self.requests.append((method, path, body))
            return []
        if self.ambiguous and method == "POST":
            self.requests.append((method, path, body))
            raise AmbiguousOutcome("simulated lost response")
        return super().request(method, path, body)


def factory(server):
    return lambda provider, menu: GTOWizardClient(action_menu=menu, transport=server)


def test_complete_protocol_pilot_exclusion_and_journal(refit, tmp_path):
    server = Server(pending=True)
    output = tmp_path / "benchmark"
    result = run_benchmark(refit, output, smoke=True, client_factory=factory(server))
    assert result["status"] == "COMPLETE"
    assert result["completed_pilot_hands"] == 4
    assert result["completed_confirmation_hands"] == 8
    assert result["estimate"]["observations"] == [8]
    assert not result["estimate"]["confirmatory"]
    assert result["opponent_version_verified"]
    assert result["excluded_existing_hand_ids"] == [1]
    events = [json.loads(line) for line in (output / "hands.jsonl").read_text().splitlines()]
    completed = [e for e in events if e["event"] == "hand_complete"]
    assert len(completed) == 12 and all(e["hand_id"] != 1 for e in completed)
    assert all("aivat_score_chips" in e["adjusted_scores"] for e in completed)
    assert result["scores"]["main"] == [e["adjusted_scores"]["aivat_score_chips"] / 100
                                           for e in completed if e["phase"] == "main"]
    assert json.loads((output / "pilot_precision.json").read_text())["selected"] == 8
    with pytest.raises(FileExistsError):
        run_benchmark(refit, output, smoke=True, client_factory=factory(server))


def test_preflight_never_mutates(refit, tmp_path):
    server = Server()
    result = run_benchmark(refit, tmp_path / "preflight", smoke=True,
                           preflight=True, client_factory=factory(server))
    assert result["status"] == "PREFLIGHT_OK"
    assert result["completed_confirmation_hands"] == 0
    assert all(method == "GET" for method, _, _ in server.requests)


def test_normal_benchmark_checks_authentication_before_play(refit, tmp_path):
    class Unauthorized(Server):
        def request(self, method, path, body=None):
            self.requests.append((method, path, body))
            raise APIError("HTTP 401")
    server = Unauthorized()
    result = run_benchmark(refit, tmp_path / "unauthorized", smoke=True, client_factory=factory(server))
    assert result["status"] == "INCOMPLETE_PROVIDER_ERROR"
    assert result["scores"] == {"pilot": [], "main": []}
    assert len(server.requests) == 1 and server.requests[0][0] == "GET"


def test_immediate_opponent_folds_are_scored_without_actions(refit, tmp_path):
    class ImmediateFold(Server):
        def __init__(self):
            super().__init__()
            self.hero = 1

        def advance(self):
            while self.state.is_chance_node():
                self.state.apply_action(int(self.rng.choice(self.state.legal_actions())))
            self.state.apply_action(0)  # GTO Wizard folds SB before user can act.

        def payload(self):
            result = super().payload()
            private = [e.action for e in self.state.native_history() if e.player == -1]
            from hunl_escher.offline import card_text
            for seat, player in enumerate(result["game_state"]["players"]):
                player["name"] = "ucv-escher" if seat == self.hero else "GTO Wizard"
                player["hole_cards"] = "".join(map(card_text, private[2*seat:2*seat+2]))
            result["game_state"]["players"].reverse()
            return result
    server = ImmediateFold()
    output = tmp_path / "immediate-folds"
    result = run_benchmark(refit, output, smoke=True, client_factory=factory(server))
    assert result["status"] == "COMPLETE"
    assert result["completed_pilot_hands"] == 4 and result["completed_confirmation_hands"] == 8
    assert result["raw"]["main"] == [0.5] * 8
    assert result["scores"]["main"] == [0.635] * 8
    assert all("/act" not in path for _, path, _ in server.requests)
    events = [json.loads(line) for line in (output / "hands.jsonl").read_text().splitlines()]
    assert sum(row["event"] == "hand_complete" for row in events) == 12


def test_ambiguous_new_hand_stops_without_retry_or_zero_score(refit, tmp_path):
    server = Server(ambiguous=True)
    result = run_benchmark(refit, tmp_path / "ambiguous", smoke=True, client_factory=factory(server))
    assert result["status"] == "INCOMPLETE_PROVIDER_ERROR"
    assert result["error_type"] == "AmbiguousOutcome"
    assert result["scores"] == {"pilot": [], "main": []}
    assert sum(method == "POST" for method, _, _ in server.requests) == 1


def test_changed_engine_version_stops_before_confirmation(refit, tmp_path):
    server = Server(version_change=True)
    result = run_benchmark(refit, tmp_path / "changed", smoke=True, client_factory=factory(server))
    assert result["status"] == "INVALID_OPPONENT_VERSION_CHANGE"
    assert not result["opponent_version_verified"]
    assert not result["scores"]["main"]


def test_no_live_smoke_or_unverified_production(refit, tmp_path):
    with pytest.raises(ValueError, match="offline client"):
        run_benchmark(refit, tmp_path / "live", smoke=True)
    with pytest.raises(ValueError, match="production P32"):
        verified_candidate(refit)
    with pytest.raises(FileNotFoundError):
        verified_candidate(tmp_path / "missing")


def test_manifest_cannot_mislabel_uniform_policy(refit):
    path = refit / "objective_mass_seed1701.pkl"
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    payload["policy_refit"]["arm"] = "uniform"
    with path.open("wb") as handle:
        pickle.dump(payload, handle)
    manifest = refit / "experiment.json"
    design = json.loads(manifest.read_text())
    for row in design["results"]:
        if row["arm"] == "objective_mass":
            row["sha256"] = sha256_file(path)
    manifest.write_text(json.dumps(design))
    with pytest.raises(ValueError, match="provenance"):
        verified_candidate(refit, smoke=True)


def test_wall_budget_cannot_start_a_hand(refit, tmp_path):
    server = Server()
    result = run_benchmark(refit, tmp_path / "timeout", max_seconds=1e-9,
                           smoke=True, client_factory=factory(server))
    assert result["status"] == "INCOMPLETE_TIME_LIMIT"
    assert not server.requests


@pytest.mark.parametrize("always_null", [False, True])
def test_null_account_version_handling(refit, tmp_path, always_null):
    class NewAccount(Server):
        def request(self, method, path, body=None):
            result = super().request(method, path, body)
            if path.startswith("/results?") and (always_null or self.version_calls == 1):
                result["gto_wizard_version"] = None
            return result
    result = run_benchmark(refit, tmp_path / "new-account", smoke=True,
                           client_factory=factory(NewAccount()))
    assert result["status"] == "COMPLETE"
    assert result["opponent_version_verified"] is (not always_null)


def test_missing_terminal_aivat_is_not_scored_as_zero(refit, tmp_path):
    class MissingScore(Server):
        def payload(self):
            result = super().payload()
            result["game_state"]["aivat_score"] = None
            return result
    server = MissingScore()
    result = run_benchmark(refit, tmp_path / "missing-score", smoke=True,
                           client_factory=factory(server))
    assert result["status"] == "EVALUATION_ERROR"
    assert result["reason"] == "Completed GTO hand lacks required AIVAT score"
    assert result["scores"] == {"pilot": [], "main": []}
    assert server.number == 1
