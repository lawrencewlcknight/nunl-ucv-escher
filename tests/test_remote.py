from copy import deepcopy
import json
from http.client import IncompleteRead

import numpy as np
import pytest
import torch

from hunl_escher.api import APIError, AmbiguousOutcome, GTOWizardClient, JSONTransport, SlumbotClient
from hunl_escher.checkpointing import LoadedHUNLPolicy, save_policy_checkpoint
from hunl_escher.config import make_solver, smoke_config
from hunl_escher.features import HUNLFeatureEncoder
from hunl_escher.game import DEFAULT_ACTION_MENU, load_hunl_game
from hunl_escher.offline import run_equivalence, synthetic_payloads
from hunl_escher.play import Agent, Journal, main, run_matches
from hunl_escher.remote import ProtocolError, parse_gtow, parse_slumbot, reconstruct


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(1)


def decision():
    s = load_hunl_game().new_initial_state()
    for card in (0, 5, 10, 15):
        s.apply_action(card)
    return s


class MockServer:
    """Independent OpenSpiel server: consumes actual API request payloads."""
    def __init__(self, provider, *, seed=1, hero=0):
        self.provider, self.rng, self.hero = provider, np.random.default_rng(seed), hero
        self.state, self.number, self.serial = None, 0, 0
        self.requests = []
        self.drop_after_mutation = False
        self.drop_before_mutation = False
        self.token = None

    def advance(self):
        while not self.state.is_terminal():
            if self.state.is_chance_node():
                self.state.apply_action(int(self.rng.choice(self.state.legal_actions())))
            elif self.state.current_player() != self.hero:
                self.state.apply_action(1)
            else:
                break

    def payload(self):
        payloads = synthetic_payloads(self.state, self.hero, hand_id=self.number)
        result = payloads[0 if self.provider == "slumbot" else 1]
        if self.provider == "slumbot":
            self.serial += 1
            self.token = f"SECRET_TOKEN_{self.serial}"
            result["token"] = self.token
        return result

    def request(self, method, path, body=None):
        self.requests.append((method, path, deepcopy(body)))
        if method == "GET":
            assert path == "/hands/in-progress?game_name=HUNL+200BB"
            return [] if self.state.is_terminal() else [self.payload()]
        if self.drop_before_mutation:
            self.drop_before_mutation = False
            raise AmbiguousOutcome("simulated uncertain request")
        if path in ("/hands", "/slumbot/api/new_hand"):
            if self.provider == "gtow":
                assert body == {"game_name": "HUNL 200BB"}
            else:
                assert body == ({} if self.token is None else {"token": self.token})
            self.number += 1
            self.state = load_hunl_game().new_initial_state()
            self.advance()
        else:
            if self.provider == "slumbot":
                assert path == "/slumbot/api/act" and body["token"] == self.token
                kind = body["incr"][0]
                amount = int(body["incr"][1:]) if kind == "b" else None
            else:
                assert path == f"/hands/{self.number}/act"
                kind, amount = body["action"], body.get("amount")
            before = self.state.decision_view()
            if kind == "b":
                self.state.apply_street_raise_to(amount)
            else:
                assert kind == "f" or kind == ("c" if before.to_call else "k")
                self.state.apply_action(0 if kind == "f" else 1)
            self.advance()
        if self.drop_after_mutation:
            self.drop_after_mutation = False
            raise AmbiguousOutcome("simulated lost response")
        return self.payload()


def test_500_offline_hands_both_protocols_all_streets():
    report = run_equivalence(500, seed=643)
    assert report["seats"] == [0, 1] and report["streets"] == [0, 1, 2, 3]
    assert report["off_menu_raises"] > 50
    assert report["adapter_observations"] == 2 * report["decisions"]
    assert report["terminal_checks"] == 2000


def test_unknown_opponent_cards_cannot_change_features():
    s = decision()
    s.apply_street_raise_to(275)
    s.apply_action(1)
    for c in (20, 25, 30):
        s.apply_action(c)
    s.apply_street_raise_to(137)
    a, b = synthetic_payloads(s, s.current_player())
    remote = parse_slumbot(a, 1)
    encoder = HUNLFeatureEncoder()
    expected = encoder.encode_decision(remote.view)
    for dummy in ((1, 2), (40, 45), (48, 49)):
        mirror = reconstruct(remote.hero, remote.hole, remote.board, remote.history, dummy_cards=dummy)
        np.testing.assert_array_equal(encoder.information_state(mirror), expected)
    b["game_state"]["players"][1 - remote.hero]["hole_cards"] = "AcAd"
    np.testing.assert_array_equal(encoder.encode_decision(parse_gtow(b, hero=remote.hero).view), expected)
    assert remote.action(4) == ("b", 961)  # native 1236, subtract 275 previous-street chips


def test_omitted_separator_and_future_board_reconstruction():
    s = decision()
    s.apply_action(1)
    s.apply_action(1)
    for c in (20, 25, 30):
        s.apply_action(c)
    a, b = synthetic_payloads(s, 1)
    assert a["action"] == "ck/"
    a["action"] = "ck"
    b["game_state"]["action_history"] = ["c", "k"]
    assert parse_slumbot(a, 1).view == s.decision_view()
    assert parse_gtow(b).view == s.decision_view()


@pytest.mark.parametrize("field,value", [("client_pos", 2), ("action", "b199"), ("action", "k"),
    ("action", "b200oops"), ("action", "/"), ("board", ["Ac"]),
    ("hole_cards", ["Ac", "Ac"]), ("hole_cards", ["XX", "Ad"]), ("winnings", 0)])
def test_reject_bad_slumbot(field, value):
    a, _ = synthetic_payloads(decision(), 0)
    a[field] = value
    with pytest.raises(ProtocolError):
        parse_slumbot(a, 1)


@pytest.mark.parametrize("field,value", [("total_pot", 151), ("street", "flop"),
    ("legal_actions", ["k"]), ("raise_range", {"min": 199, "max": 20000}),
    ("action_history", ["b275", "bad"]), ("player_to_act", "gto_wizard"),
    ("is_hand_over", True)])
def test_reject_gtow_disagreement(field, value):
    _, b = synthetic_payloads(decision(), 0)
    b["game_state"][field] = value
    with pytest.raises(ProtocolError):
        parse_gtow(b)


def test_allin_runout_and_short_allin():
    s = decision()
    s.apply_native_action(10100)
    s.apply_native_action(20000)
    a, b = synthetic_payloads(s, 0)
    assert parse_slumbot(a, 1).view.legal_mask == (1, 1, 0, 0, 0, 0, 0)
    assert parse_gtow(b).view == s.decision_view()
    s.apply_action(1)
    while s.is_chance_node():
        s.apply_action(s.chance_outcomes()[0][0])
    a, b = synthetic_payloads(s, 0)
    assert a["action"].endswith("///")
    assert parse_slumbot(a, 1).terminal and parse_gtow(b).terminal


@pytest.fixture
def checkpoint(tmp_path):
    s = make_solver(smoke_config(), seed=47)
    s.iteration()
    s.train_average_policy()
    return save_policy_checkpoint(s, tmp_path / "policy.pkl", seed=47, config=smoke_config(), checkpoint_row={})


def test_saved_policy_same_probabilities_through_both_adapters(checkpoint):
    policy = LoadedHUNLPolicy(load_hunl_game(), checkpoint)
    report = run_equivalence(30, seed=47, policy=policy)
    assert report["policy_probability_checks"] == report["adapter_observations"]
    # Inference-only object does not construct trainers or replay.
    assert not hasattr(policy, "q_value_trainer") and not hasattr(policy, "buffer")
    agent1 = Agent("checkpoint", checkpoint=checkpoint, seed=32)
    agent2 = Agent("checkpoint", checkpoint=checkpoint, seed=32)
    view = decision().decision_view()
    assert [agent1.choose(view) for _ in range(30)] == [agent2.choose(view) for _ in range(30)]


@pytest.mark.parametrize("provider", ["slumbot", "gtow"])
@pytest.mark.parametrize("hero", [0, 1])
def test_complete_mock_sessions_and_secret_free_journal(tmp_path, provider, hero):
    server = MockServer(provider, hero=hero)
    cls = SlumbotClient if provider == "slumbot" else GTOWizardClient
    client = cls(transport=server)
    path = tmp_path / "run.jsonl"
    log = Journal(path)
    try:
        summary = run_matches(client, Agent("check_call"), hands=8, journal=log)
    finally:
        log.close()
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    completed = [r for r in rows if r["event"] == "hand_complete"]
    assert summary["completed_hands"] == len(completed) == 8
    assert summary["winnings_chips"] == sum(r["winnings_chips"] for r in completed)
    assert all(len(r["board"]) == 5 for r in completed)
    assert all(r["adjusted_scores"] for r in completed)
    assert "SECRET_TOKEN" not in path.read_text()
    with pytest.raises(FileExistsError):
        Journal(path)


def test_gtow_recovers_applied_action_without_resending():
    server = MockServer("gtow")
    client = GTOWizardClient(transport=server)
    client.new_hand()
    server.drop_after_mutation = True
    after = client.act(2)
    assert after.view.street == 1 and client.recoveries == 1
    assert [m for m, p, b in server.requests] == ["POST", "POST", "GET"]


def test_gtow_unchanged_after_timeout_is_not_retried():
    server = MockServer("gtow")
    client = GTOWizardClient(transport=server)
    client.new_hand()
    server.drop_before_mutation = True
    with pytest.raises(AmbiguousOutcome):
        client.act(2)
    assert [m for m, p, b in server.requests] == ["POST", "POST", "GET"]
    with pytest.raises(AmbiguousOutcome, match="Session stopped"):
        client.act(2)
    assert len(server.requests) == 3


def test_gtow_missing_finished_hand_after_timeout_not_counted_as_success():
    server = MockServer("gtow")
    client = GTOWizardClient(transport=server)
    client.new_hand()
    server.drop_after_mutation = True
    with pytest.raises(AmbiguousOutcome, match="terminal outcome unresolved"):
        client.act(6)
    assert len(server.requests) == 3


def test_slumbot_timeout_stops_and_logs_no_duplicate_action(tmp_path):
    server = MockServer("slumbot")
    server.drop_after_mutation = True  # even new_hand may have succeeded
    path = tmp_path / "ambiguous.jsonl"
    log = Journal(path)
    try:
        with pytest.raises(AmbiguousOutcome):
            run_matches(SlumbotClient(transport=server), Agent(), hands=3, journal=log)
    finally:
        log.close()
    assert len(server.requests) == 1
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[-1]["event"] == "stopped" and rows[-1]["completed_hands"] == 0


@pytest.mark.parametrize("failure", [TimeoutError("SECRET_KEY_SHOULD_NOT_BE_LOGGED"), IncompleteRead(b"SECRET_PARTIAL_BODY")])
def test_http_get_backoff_but_never_post_retry(monkeypatch, failure):
    sleeps = []
    transport = JSONTransport("https://example.invalid", sleep=sleeps.append)
    attempts = []
    def broken(*args):
        attempts.append(args)
        raise failure
    monkeypatch.setattr(transport, "_once", broken)
    with pytest.raises(AmbiguousOutcome) as exc:
        transport.request("POST", "/hands", {})
    assert len(attempts) == 1 and sleeps == [] and "SECRET" not in str(exc.value)
    attempts.clear()
    with pytest.raises(APIError):
        transport.request("GET", "/hands/in-progress")
    assert len(attempts) == 3 and sleeps == [1, 2]


def test_cli_offline_no_credentials_and_existing_log_guard(tmp_path, monkeypatch):
    monkeypatch.delenv("GTO_WIZARD_API_KEY", raising=False)
    path = tmp_path / "offline.jsonl"
    args = ["--opponent", "offline", "--hands", "5", "--output", str(path)]
    main(args)
    assert json.loads(path.read_text().splitlines()[-1])["kind"] == "synthetic_offline_equivalence"
    with pytest.raises(FileExistsError):
        main(args)
    with pytest.raises(SystemExit):
        main(["--opponent", "gtow", "--output", str(tmp_path / "gto.jsonl")])
    assert not (tmp_path / "gto.jsonl").exists()


def test_later_board_can_use_a_previous_placeholder_card():
    s = decision()
    a, _ = synthetic_payloads(s, 0)
    before = parse_slumbot(a, 1)
    # Initially the shadow opponent gets card 1, which is not a real observation.
    s.apply_action(1)
    s.apply_action(1)
    for card in (1, 20, 25):
        s.apply_action(card)
    s.apply_action(1)  # real opponent checks to hero
    a, _ = synthetic_payloads(s, 0)
    after = parse_slumbot(a, 1)
    assert 1 in after.board and after.view == s.decision_view()
    assert before.hole == after.hole


def test_gto_resume_reads_existing_hand_without_creating_one():
    server = MockServer("gtow", hero=1)
    GTOWizardClient(transport=server).new_hand()
    client = GTOWizardClient(transport=server)
    result = client.resume_hand(1)
    assert result.hero == 1 and result.hand_id == 1
    assert [m for m, p, b in server.requests] == ["POST", "GET"]


def test_http_auth_errors_are_not_retried_and_hide_response(monkeypatch):
    from urllib.error import HTTPError
    t = JSONTransport("https://example.invalid")
    calls = []
    def unauthorized(*args):
        calls.append(args)
        raise HTTPError("https://example.invalid", 401, "SECRET_SERVER_MESSAGE", {}, None)
    monkeypatch.setattr(t, "_once", unauthorized)
    with pytest.raises(APIError) as exc:
        t.request("POST", "/hands", {})
    assert str(exc.value) == "HTTP 401" and len(calls) == 1
