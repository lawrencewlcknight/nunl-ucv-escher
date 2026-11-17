import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from hunl_escher.api import APIError, AmbiguousOutcome, GTOWizardClient, SlumbotClient
from hunl_escher.benchmark import run_benchmark
from hunl_escher.checkpointing import load_checkpoint_payload
from hunl_escher.evaluate import duplicate_pair
from hunl_escher.evaluation_stats import estimate, sample_plan
from hunl_escher.game import load_hunl_game
from hunl_escher.play import Agent
from test_remote import MockServer


def test_duplicate_deals_exactly_cancel_identical_deterministic_agents():
    game = load_hunl_game()
    for pair in range(20):
        result = duplicate_pair(game, [Agent(), Agent()], 123, pair)
        assert sum(result) == 0


def test_duplicate_sampling_reproducible_and_zero_sum_under_swap():
    game = load_hunl_game()
    a = duplicate_pair(game, [Agent("random"), Agent("check_call")], 51, 4)
    b = duplicate_pair(game, [Agent("random"), Agent("check_call")], 51, 4)
    assert a == b and all(abs(x) <= 200 for x in a)
    # Deterministic opposing policies give exact sign reversal on swapping A/B.
    x = duplicate_pair(game, [Agent("all_in"), Agent("check_call")], 51, 4)
    y = duplicate_pair(game, [Agent("check_call"), Agent("all_in")], 51, 4)
    assert sum(x) == -sum(y)


def test_statistical_units_power_and_incomplete_claims():
    pilot = [-20., 20.]*5000
    plan = sample_plan(pilot, minimum=100000, maximum=1000000)
    assert 313000 < plan["selected"] < 315000
    independent = sample_plan(pilot, other=pilot, minimum=2, maximum=1000000)
    assert abs(independent["required"]-2*plan["required"]) <= 1
    capped = sample_plan(pilot, minimum=2, maximum=1000)
    assert capped["capped"] and capped["estimated_detectable_effect_bb100"] > 10
    assert estimate([1., 2.]*100)["conclusion"] == "improved"
    assert estimate([1., 2.]*100, complete=False)["confirmatory"] is False
    assert estimate([1., 1.]*100)["confirmatory"] is False
    result = estimate([2., 3.]*100, other=[1., 2.]*100, alpha=.025)
    assert result["bb_per_100"] == 100 and result["confidence"] == .975


def test_report_does_not_claim_strength_from_smoke(tmp_path):
    from hunl_escher.experiment_report import render
    payload = dict(status="TRAINING_COMPLETE", smoke=True,
                   evaluations={"primary": {"status": "COMPLETE", "result": estimate([1., 2.]*100)}})
    (tmp_path / "experiment.json").write_text(json.dumps(payload))
    report = render(tmp_path)
    assert report["strength_evidence"].startswith("SMOKE ONLY")
    payload["smoke"] = False
    payload["evaluations"]["primary"]["result"] = estimate([-1., -2.]*100)
    (tmp_path / "experiment.json").write_text(json.dumps(payload))
    assert "regression" in render(tmp_path)["strength_evidence"]


@pytest.fixture(scope="module")
def experiment(tmp_path_factory):
    output = tmp_path_factory.mktemp("learning") / "run"
    command = [sys.executable, "-m", "hunl_escher.experiment", "--output", str(output),
               "--smoke", "--stage-seconds", ".001", "--training-wall-seconds", "180",
               "--evaluation-seconds", "180"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=240)
    assert result.returncode == 0, result.stdout[-2000:]+result.stderr[-5000:]
    return output


def test_real_restart_and_playable_checkpoints(experiment):
    report = json.loads((experiment / "experiment.json").read_text())
    assert report["status"] == "TRAINING_COMPLETE"
    a, b = [load_checkpoint_payload(experiment / f"{label}.pkl") for label in ("P4", "P8")]
    assert a["outer_iteration"] == 1 and b["outer_iteration"] == 2
    assert b["training_elapsed_seconds"] > a["training_elapsed_seconds"]
    assert len(list((experiment / "training").glob("state_*"))) == 1
    primary = report["evaluations"]["primary"]
    assert primary["status"] == "COMPLETE"
    assert primary["completed_pairs"] == 8
    assert len((experiment / "primary" / "pilot.jsonl").read_text().splitlines()) == 4
    assert all(item["status"] == "COMPLETE" for item in report["evaluations"].values())
    assert report["training_diagnostics"]["finite_exported_policy_weights"]
    assert report["training_diagnostics"]["finite_recorded_metrics"]
    assert report["training_diagnostics"]["exported_policy_changed"]
    assert (experiment / "learning_report.md").exists()


def test_multiprocess_evaluation_is_deterministic(experiment, tmp_path):
    from hunl_escher.evaluate import run_evaluation
    a, b = experiment / "P8.pkl", str(experiment / "P4.pkl")
    for workers in (1, 2):
        report = run_evaluation(a, b, tmp_path / str(workers), workers=workers,
                                pilot_pairs=4, minimum=8, maximum=8, max_seconds=60)
        assert report["status"] == "COMPLETE"
    for filename in ("pilot.jsonl", "confirmatory.jsonl"):
        assert (tmp_path / "1" / filename).read_bytes() == (tmp_path / "2" / filename).read_bytes()


def test_cloud_plan_is_bounded_and_source_archive_excludes_secrets(tmp_path):
    import importlib.util
    import tarfile
    script = Path(__file__).resolve().parents[1] / "tools" / "launch_learning_check.py"
    spec = importlib.util.spec_from_file_location("launch", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    plan = module.build_bundle(tmp_path / "plan", project="test-project", zone="us-central1-a",
                bucket="gs://test-bucket/hunl", service_account="runner@test-project.iam.gserviceaccount.com",
                run_name="test-run", gto_secret="optional-key")
    assert len(plan["commands"]) == 3 and plan["training_active_hours"] == 8
    assert "15h" in plan["commands"][1] and "26h" in plan["commands"][2]
    for command in plan["commands"][1:]:
        assert "--no-restart-on-failure" in command and "STOP" in command
    with tarfile.open(tmp_path / "plan" / "source.tar.gz") as archive:
        names = archive.getnames()
    assert "hunl_escher/experiment.py" in names
    assert not any(".env" in name or ".git" in name or "outputs/" in name for name in names)
    for role in ("train", "external"):
        subprocess.run(["bash", "-n", str(tmp_path / "plan" / f"{role}-startup.sh")], check=True)


def test_missing_gtow_key_skips_without_waiting_or_loading_policy(tmp_path, monkeypatch):
    monkeypatch.delenv("GTO_WIZARD_API_KEY", raising=False)
    result = run_benchmark("gtow", tmp_path / "absent", tmp_path / "out")
    assert result["status"] == "SKIPPED_UNAVAILABLE"


@pytest.mark.parametrize("provider", ["slumbot", "gtow"])
def test_benchmark_full_protocol_offline(experiment, tmp_path, provider):
    class Transport(MockServer):
        def request(self, method, path, body=None):
            if path.startswith("/results?"):
                return {"gto_wizard_version": 1}
            return super().request(method, path, body)
    def factory(name, menu):
        cls = SlumbotClient if name == "slumbot" else GTOWizardClient
        return cls(action_menu=menu, transport=Transport(name))
    result = run_benchmark(provider, experiment, tmp_path / "out", pilot_hands=4,
                           minimum=8, maximum=8, block_size=2, client_factory=factory)
    assert result["status"] == "COMPLETE"
    assert result["improvement"]["confidence"] == .975
    assert all(len(result["scores"][label]["main"]) == 8 for label in ("P4", "P8"))
    journal = (tmp_path / "out" / "hands.jsonl").read_text()
    assert "SECRET_TOKEN" not in journal


@pytest.mark.parametrize("error,status", [(APIError("HTTP 403"), "SKIPPED_UNAVAILABLE"),
                          (AmbiguousOutcome("uncertain"), "INCOMPLETE_PROVIDER_ERROR")])
def test_provider_failure_isolated(experiment, tmp_path, error, status):
    def factory(*args):
        raise error
    result = run_benchmark("gtow", experiment, tmp_path / "out", client_factory=factory)
    assert result["status"] == status and result["improvement"] is None


def test_opponent_version_change_invalidates_main_sample(experiment, tmp_path):
    class Transport(MockServer):
        versions = 0
        def request(self, method, path, body=None):
            if path.startswith("/results?"):
                self.versions += 1
                return {"gto_wizard_version": self.versions}
            return super().request(method, path, body)
    def factory(name, menu):
        return GTOWizardClient(action_menu=menu, transport=Transport(name))
    result = run_benchmark("gtow", experiment, tmp_path / "version", pilot_hands=2,
                           minimum=2, maximum=2, client_factory=factory)
    assert result["status"] == "INVALID_OPPONENT_VERSION_CHANGE"
    assert result["improvement"]["confirmatory"] is False


def test_interrupted_main_never_claims_significance(experiment, tmp_path):
    class Transport(MockServer):
        def request(self, method, path, body=None):
            if self.number == 10 and path == "/slumbot/api/new_hand":
                raise AmbiguousOutcome("Lost new-hand response")
            return super().request(method, path, body)
    def factory(name, menu):
        return SlumbotClient(action_menu=menu, transport=Transport(name))
    result = run_benchmark("slumbot", experiment, tmp_path / "partial", pilot_hands=2,
                           minimum=8, maximum=8, block_size=2, client_factory=factory)
    assert result["status"] == "INCOMPLETE_PROVIDER_ERROR"
    assert sum(len(result["scores"][label]["main"]) for label in ("P4", "P8")) == 6
    assert result["improvement"]["confirmatory"] is False


def test_cloud_recovery_publication_prunes_only_after_success(tmp_path, monkeypatch):
    from hunl_escher.experiment import upload_recovery
    training = tmp_path / "training"
    training.mkdir()
    (training / "latest.json").write_text(json.dumps({"state": "state_000002"}))
    (tmp_path / "cloud_checkpoint.json").write_text(json.dumps({"state": "state_000001"}))
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout='gs://bucket/run/training/state_000001/\ngs://bucket/run/training/state_000002/\n', stderr='')
    monkeypatch.setattr(subprocess, "run", run)
    upload_recovery(tmp_path, "gs://bucket/run")
    assert commands[0][2] == "rsync"
    assert commands[2][-1] == "gs://bucket/run/training/latest.json"
    assert commands[-1] == ["gcloud", "storage", "rm", "--recursive", "gs://bucket/run/training/state_000001"]
    (tmp_path / "cloud_checkpoint.json").write_text(json.dumps({"state": "state_000001"}))
    commands.clear()
    def fail(command, **kwargs):
        commands.append(command)
        raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        upload_recovery(tmp_path, "gs://bucket/run")
    assert len(commands) == 1
    assert json.loads((tmp_path / "cloud_checkpoint.json").read_text())["state"] == "state_000001"


def test_invalid_active_budget_rejected_before_training(tmp_path):
    for value in ("0", "-1", "nan", "inf"):
        result = subprocess.run([sys.executable, "-m", "hunl_escher.train", "--smoke",
                  "--output", str(tmp_path / "unused"), "--additional-training-seconds", value],
                  capture_output=True, text=True, timeout=30)
        assert result.returncode != 0
    assert not (tmp_path / "unused").exists()
