"""Independent fixed-sample external benchmarks of the verified HUNL P128 policy."""
import argparse
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from .api import APIError, JSONTransport, SlumbotClient
from .benchmark import client_for, collect_hand
from .checkpointing import load_checkpoint_payload, sha256_file
from .evaluation_stats import estimate, sample_plan
from .play import Agent, Journal
from .continuation_evaluate import verified_checkpoint
from .gtow_p32 import account_snapshot, DeadlineTransport
from .remote import ProtocolError, parse_gtow
from .train import write_json

POLICY_SHA256 = "aade5f09a22fc7aeddc7063e4250fc00f04c16cc5c4827381f775e3d66727bb3"
MANIFEST_SHA256 = "a08210f79bcddebc564109886673d38cb7c8002326b682b2b66f085eb54c048b"
SOURCE_URI = "gs://clever-overview-399515-fhp-escher-results/hunl/hunl-p104-plus24-batch-20261122"
LABEL = "P128"
COUNTS = {"slumbot": (0, 20000), "gtow": (0, 8000)}


def verified_candidate(experiment, *, smoke=False):
    root = Path(experiment)
    policy, marker = verified_checkpoint(root, LABEL, smoke=smoke)
    manifest_hash = sha256_file(root / "continuation.json")
    if not smoke and (marker["sha256"] != POLICY_SHA256
                      or marker["outer_iteration"] != 6514 or manifest_hash != MANIFEST_SHA256):
        raise ValueError("Requires the pinned production P128 training run")
    payload = load_checkpoint_payload(policy)
    if payload["outer_iteration"] != marker["outer_iteration"]:
        raise ValueError("Policy iteration differs from training marker")
    if not smoke:
        training = json.loads((root / "continuation.json").read_text())
        validation = training["checkpoints"][LABEL]["metrics"]["policy_validation"]
        if validation["group_sampling"] != "objective_mass":
            raise ValueError("P128 must use the corrected policy grouping sampler")
    return dict(kind="checkpoint", checkpoint=policy), dict(policy_sha256=marker["sha256"],
        source_manifest_sha256=manifest_hash, outer_iteration=marker["outer_iteration"],
        active_seconds=marker["active_seconds"])


def benchmark_client(provider, menu):
    if provider == "slumbot":
        return SlumbotClient(action_menu=menu, transport=JSONTransport(
            "https://slumbot.com", timeout=180, min_interval=.5))
    return client_for(provider, menu)


class HandBudget:
    """Reserve each new-hand attempt, including any unresolved final attempt.

    No uncertain POST is replaced by a fresh hand. GTO's documented engine-busy
    retries remain internal to its transport; they retry the same request only.
    """
    def __init__(self, client, maximum):
        self.client, self.maximum, self.attempted = client, maximum, 0

    def __getattr__(self, name):
        return getattr(self.client, name)

    def new_hand(self):
        if self.attempted >= self.maximum:
            raise ProtocolError("Fresh hand budget exhausted")
        self.attempted += 1
        return self.client.new_hand()


def run_benchmark(experiment, output, *, provider, max_seconds=86400, seed=None,
                  preflight=False, smoke=False, client_factory=None):
    if provider not in COUNTS:
        raise ValueError("Unknown benchmark provider")
    if seed is None:
        seed = 1287901 if provider == "slumbot" else 1288901
    if not math.isfinite(max_seconds) or max_seconds <= 0 or seed < 0:
        raise ValueError("Invalid execution limits")
    if smoke and client_factory is None:
        raise ValueError("Synthetic smoke requires an injected offline client")
    spec, identity = verified_candidate(experiment, smoke=smoke)
    if provider == "gtow" and client_factory is None and not os.environ.get("GTO_WIZARD_API_KEY"):
        raise ValueError("Set GTO_WIZARD_API_KEY in the environment")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    pilot_count, main_count = (0, 8) if smoke else COUNTS[provider]
    design = dict(provider=provider, policy=LABEL, game_name="HUNL 200BB", **identity,
        seed=seed, pilot_hands=pilot_count, confirmation_hands=main_count,
        pilot_excluded=True, fixed_sample=True, alpha=.05, max_seconds=max_seconds,
        primary_metric="aivat_bb" if provider == "gtow" else "raw_bb", secondary_metric="raw_bb", smoke=smoke,
        total_hand_cap=pilot_count + main_count, request_timeout_seconds=180, min_request_interval_seconds=.5,
        preflight_only=preflight, interval_unit="independent fresh server hand",
        interpretation="Absolute benchmark of one fixed trained policy; not an improvement or exploitability test",
        implementation_sha256={name: sha256_file(Path(__file__).with_name(name))
            for name in ("p128_benchmark.py", "gtow_p32.py", "api.py", "remote.py", "play.py", "benchmark.py")})
    write_json(output / "design.json", design)
    report = dict(status="STARTING", **design, scores={"pilot": [], "main": []},
                  raw={"pilot": [], "main": []}, version_checks=[], estimate=None,
                  raw_estimate=None, opponent_version_verified=False)
    write_json(output / "report.json", report)
    journal = Journal(output / "hands.jsonl")
    deadline = time.monotonic() + max_seconds

    def check_version(client):
        if provider != "gtow":
            return
        snapshot = account_snapshot(client)
        snapshot.update(completed_pilot_hands=len(report["scores"]["pilot"]),
                        completed_confirmation_hands=len(report["scores"]["main"]))
        report["version_checks"].append(snapshot)
        known = {x["gto_wizard_version"] for x in report["version_checks"]
                 if x["gto_wizard_version"] is not None}
        if len(known) > 1:
            raise ProtocolError("Opponent engine version changed")

    client = None
    hand_in_flight = False
    completed_ids = set()
    try:
        import torch
        torch.set_num_threads(1)
        agent = Agent(**spec, seed=seed)
        if agent.checkpoint_hash != identity["policy_sha256"]:
            raise ValueError("Policy changed after verification")
        client = (client_factory or benchmark_client)(provider, agent.action_menu)
        if client.provider != provider:
            raise ValueError("Unexpected benchmark client")
        client.transport = DeadlineTransport(client.transport, deadline)
        # Existing account hands have unknown policy provenance. Validate them,
        # but never resume them or include them in this experiment's results.
        pending = client._in_progress() if provider == "gtow" else []
        report["excluded_existing_hand_ids"] = [parse_gtow(p, agent.action_menu).hand_id for p in pending]
        completed_ids.update(report["excluded_existing_hand_ids"])
        client = HandBudget(client, pilot_count + main_count)
        check_version(client)
        journal.write("policy", **identity, seed=seed, action_menu=list(agent.action_menu))
        if preflight:
            report["status"] = "PREFLIGHT_OK"
            return report
        for phase, count in (("pilot", pilot_count), ("main", main_count)):
            if count == 0:
                continue
            agent.rng = np.random.default_rng(seed + (phase == "main"))
            report["status"] = phase.upper()
            write_json(output / "report.json", report)
            for ordinal in range(count):
                if time.monotonic() >= deadline:
                    raise TimeoutError("Benchmark wall budget exhausted")
                hand_in_flight = True
                score, raw = collect_hand(client, agent, journal, phase, LABEL, ordinal)
                if not math.isfinite(score) or not math.isfinite(raw):
                    raise ProtocolError("Non-finite terminal score")
                hand_id = client.current.hand_id
                if hand_id in completed_ids:
                    raise ProtocolError("Duplicate or previously excluded hand identity")
                completed_ids.add(hand_id)
                report["scores"][phase].append(score)
                report["raw"][phase].append(raw)
                hand_in_flight = False
                if (ordinal + 1) % 100 == 0:
                    check_version(client)
                if (ordinal + 1) % 25 == 0:
                    report["fresh_hand_attempts"] = client.attempted
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
                    if r["completed_pilot_hands"] == pilot_count and r["completed_confirmation_hands"] > 0]
        known = {r["gto_wizard_version"] for r in report["version_checks"]
                 if r["gto_wizard_version"] is not None}
        report["opponent_version_verified"] = bool(versions) and None not in versions and len(known) == 1
        report["fresh_hand_attempts"] = client.attempted if isinstance(client, HandBudget) else 0
        report["unresolved_hand_attempt"] = hand_in_flight
        report["incomplete_sample_warning"] = (None if report["status"] == "COMPLETE" else
            "Missing or unfinished hands may bias partial scores; do not treat these as a completed benchmark")
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
    parser.add_argument("--experiment", required=True, type=Path, help="Directory containing verified P128.pkl, P128.json and continuation.json")
    parser.add_argument("--output", required=True, type=Path, help="New immutable benchmark output directory")
    parser.add_argument("--max-seconds", type=float, default=86400)
    parser.add_argument("--provider", required=True, choices=tuple(COUNTS))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--preflight", action="store_true", help="Verify policy and authenticated GETs; play no hands")
    report = run_benchmark(**vars(parser.parse_args(argv)))
    print(json.dumps({k: report[k] for k in ("status", "policy_sha256", "completed_confirmation_hands", "estimate")}, indent=2))
    if report["status"] not in ("COMPLETE", "PREFLIGHT_OK"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
