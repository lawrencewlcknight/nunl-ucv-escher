"""Fixed-sample GTO Wizard benchmark of the objective-mass P32 refit."""
import argparse
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from .api import APIError, JSONTransport
from .benchmark import client_for, collect_hand
from .checkpointing import load_checkpoint_payload, sha256_file
from .evaluation_stats import estimate, sample_plan
from .play import Agent, Journal
from .policy_sampling_evaluate import verified_refits
from .remote import ProtocolError, parse_gtow
from .train import write_json

P32_SHA256 = "73eec96257ba92f0c6b8e4fb43f040d8f88a7eb80c589dcffd71f567bb49f129"
FIT_SEED = 1701
LABEL = "P32_objective_mass_seed1701"


def verified_candidate(experiment, *, smoke=False):
    """Reject original P32, incomplete fits and post-hoc seed selection."""
    design, specs, hashes = verified_refits(experiment)
    seed = design["external_fit_seed"]
    if not smoke and (design["smoke"] or design["source_policy_sha256"] != P32_SHA256
                      or design["source_iteration"] != 1658 or seed != FIT_SEED):
        raise ValueError("Requires a production P32 objective-mass refit with predeclared seed 1701")
    name = f"objective_mass_{seed}"
    spec = specs[name]
    payload = load_checkpoint_payload(spec["checkpoint"])
    refit = payload.get("policy_refit", {})
    if (payload.get("checkpoint_kind") != "frozen_replay_policy_refit"
            or refit.get("arm") != "objective_mass" or refit.get("fit_seed") != seed
            or refit.get("source_sha256") != design["source_policy_sha256"]
            or refit.get("smoke") != design["smoke"]
            or payload["outer_iteration"] != design["source_iteration"]):
        raise ValueError("Corrected checkpoint provenance does not match the refit manifest")
    return spec, dict(policy_sha256=hashes[name], source_policy_sha256=design["source_policy_sha256"],
                     fit_seed=seed, arm="objective_mass",
                     refit_manifest_sha256=sha256_file(Path(experiment) / "experiment.json"))


def account_snapshot(client):
    """Only retain score metadata, never credentials or arbitrary server fields."""
    data = client.transport.request("GET", "/results?game_name=HUNL+200BB")
    if not isinstance(data, dict):
        raise ProtocolError("Malformed account results")
    version = data.get("gto_wizard_version")
    if version is not None and (type(version) is not int or version < 1):
        raise ProtocolError("Invalid opponent version")
    return dict(gto_wizard_version=version, total_hands=data.get("total_hands"))


class DeadlineTransport:
    """Bound every request, including decisions within an unfinished hand."""
    def __init__(self, transport, deadline):
        self.transport, self.deadline = transport, deadline
        if isinstance(transport, JSONTransport):
            transport.deadline = deadline  # Also bound engine-busy retry waits.

    def request(self, *args, **kwargs):
        if time.monotonic() >= self.deadline:
            raise TimeoutError("Benchmark wall budget exhausted")
        return self.transport.request(*args, **kwargs)


def run_benchmark(experiment, output, *, max_seconds=86400, seed=671901,
                  preflight=False, smoke=False, client_factory=None):
    if not math.isfinite(max_seconds) or max_seconds <= 0 or seed < 0:
        raise ValueError("Invalid execution limits")
    if smoke and client_factory is None:
        raise ValueError("Synthetic smoke requires an injected offline client")
    spec, identity = verified_candidate(experiment, smoke=smoke)
    if client_factory is None and not os.environ.get("GTO_WIZARD_API_KEY"):
        raise ValueError("Set GTO_WIZARD_API_KEY in the environment")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    pilot_count, main_count = (4, 8) if smoke else (1000, 20000)
    design = dict(provider="gtow", policy=LABEL, game_name="HUNL 200BB", **identity,
        seed=seed, pilot_hands=pilot_count, confirmation_hands=main_count,
        pilot_excluded=True, fixed_sample=True, alpha=.05, max_seconds=max_seconds,
        primary_metric="aivat_bb", secondary_metric="raw_bb", smoke=smoke,
        preflight_only=preflight, interval_unit="independent fresh server hand",
        interpretation="Absolute benchmark of one fixed refit; not an improvement or exploitability test",
        implementation_sha256={name: sha256_file(Path(__file__).with_name(name))
            for name in ("gtow_p32.py", "api.py", "remote.py", "play.py", "benchmark.py")})
    write_json(output / "design.json", design)
    report = dict(status="STARTING", **design, scores={"pilot": [], "main": []},
                  raw={"pilot": [], "main": []}, version_checks=[], estimate=None,
                  raw_estimate=None, opponent_version_verified=False)
    write_json(output / "report.json", report)
    journal = Journal(output / "hands.jsonl")
    deadline = time.monotonic() + max_seconds

    def check_version(client):
        snapshot = account_snapshot(client)
        snapshot.update(completed_pilot_hands=len(report["scores"]["pilot"]),
                        completed_confirmation_hands=len(report["scores"]["main"]))
        report["version_checks"].append(snapshot)
        known = {x["gto_wizard_version"] for x in report["version_checks"]
                 if x["gto_wizard_version"] is not None}
        if len(known) > 1:
            raise ProtocolError("Opponent engine version changed")

    try:
        import torch
        torch.set_num_threads(1)
        agent = Agent(**spec, seed=seed)
        if agent.checkpoint_hash != identity["policy_sha256"]:
            raise ValueError("Policy changed after verification")
        client = (client_factory or client_for)("gtow", agent.action_menu)
        if client.provider != "gtow":
            raise ValueError("Expected GTO Wizard client")
        client.transport = DeadlineTransport(client.transport, deadline)
        # Existing account hands have unknown policy provenance. Validate them,
        # but never resume them or include them in this experiment's results.
        pending = client._in_progress()
        report["excluded_existing_hand_ids"] = [parse_gtow(p, agent.action_menu).hand_id for p in pending]
        check_version(client)
        journal.write("policy", **identity, seed=seed, action_menu=list(agent.action_menu))
        if preflight:
            report["status"] = "PREFLIGHT_OK"
            return report
        for phase, count in (("pilot", pilot_count), ("main", main_count)):
            agent.rng = np.random.default_rng(seed + (phase == "main"))
            report["status"] = phase.upper()
            write_json(output / "report.json", report)
            for ordinal in range(count):
                if time.monotonic() >= deadline:
                    raise TimeoutError("Benchmark wall budget exhausted")
                score, raw = collect_hand(client, agent, journal, phase, LABEL, ordinal)
                if not math.isfinite(score) or not math.isfinite(raw):
                    raise ProtocolError("Non-finite terminal score")
                report["scores"][phase].append(score)
                report["raw"][phase].append(raw)
                if (ordinal + 1) % 100 == 0:
                    check_version(client)
                    write_json(output / "report.json", report)
            check_version(client)
            if phase == "pilot":
                plan = sample_plan(report["scores"]["pilot"], minimum=main_count, maximum=main_count)
                plan["interpretation"] = "Precision diagnostic only; confirmation count was fixed before pilot"
                write_json(output / "pilot_precision.json", plan)
        report["status"] = "COMPLETE"
    except TimeoutError:
        report["status"] = "INCOMPLETE_TIME_LIMIT"
    except APIError as exc:
        report.update(status="INCOMPLETE_PROVIDER_ERROR", error_type=type(exc).__name__, reason=str(exc))
    except (ProtocolError, ValueError, RuntimeError) as exc:
        report.update(status="EVALUATION_ERROR", error_type=type(exc).__name__)
        if isinstance(exc, ProtocolError):
            # Adapter errors are locally defined diagnostics, not raw HTTP bodies.
            report["reason"] = str(exc)
        if str(exc) == "Opponent engine version changed":
            report["status"] = "INVALID_OPPONENT_VERSION_CHANGE"
    except BaseException:
        report["status"] = "INTERRUPTED"
        raise
    finally:
        # A new account can report null before its first completed hand. Use
        # the check immediately after the pilot and all confirmation checks.
        versions = [r["gto_wizard_version"] for r in report["version_checks"]
                    if r["completed_pilot_hands"] == pilot_count]
        known = {r["gto_wizard_version"] for r in report["version_checks"]
                 if r["gto_wizard_version"] is not None}
        report["opponent_version_verified"] = bool(versions) and None not in versions and len(known) == 1
        report["completed_confirmation_hands"] = len(report["scores"]["main"])
        report["completed_pilot_hands"] = len(report["scores"]["pilot"])
        for field, source in (("estimate", "scores"), ("raw_estimate", "raw")):
            values = report[source]["main"]
            if len(values) >= 2:
                result = estimate(values, complete=False)
                result.update(conclusion="descriptive_absolute_benchmark", confirmatory=False,
                    full_fixed_sample=report["status"] == "COMPLETE" and len(values) == main_count)
                report[field] = result
        journal.close()
        write_json(output / "report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, type=Path, help="Completed frozen-replay refit directory")
    parser.add_argument("--output", required=True, type=Path, help="New immutable benchmark output directory")
    parser.add_argument("--max-seconds", type=float, default=86400)
    parser.add_argument("--seed", type=int, default=671901)
    parser.add_argument("--preflight", action="store_true", help="Verify corrected policy and authenticated GETs; play no hands")
    report = run_benchmark(**vars(parser.parse_args(argv)))
    print(json.dumps({k: report[k] for k in ("status", "policy_sha256", "completed_confirmation_hands", "estimate")}, indent=2))
    if report["status"] not in ("COMPLETE", "PREFLIGHT_OK"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
