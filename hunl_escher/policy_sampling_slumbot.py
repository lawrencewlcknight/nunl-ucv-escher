"""Separately invoked fresh-arm Slumbot follow-up after the local primary passes."""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.stats import norm

from .api import APIError
from .benchmark import client_for, collect_hand, version_for
from .checkpointing import sha256_file
from .evaluation_stats import estimate
from .play import Agent, Journal
from .policy_sampling_evaluate import verified_refits
from .remote import ProtocolError
from .train import write_json

ARMS = ("uniform", "objective_mass")


def verified_gate(experiment, evaluation, *, smoke=False):
    design, specs, hashes = verified_refits(experiment)
    root = Path(evaluation)
    manifest = json.loads((root / "manifest.json").read_text())
    report = json.loads((root / "report.json").read_text())
    if (manifest["experiment_sha256"] != sha256_file(Path(experiment) / "experiment.json")
            or manifest["policies"] != hashes or manifest["fit_seeds"] != design["fit_seeds"]
            or manifest["external_fit_seed"] != design["external_fit_seed"]):
        raise ValueError("Local evaluation does not match the frozen experiment")
    result = report.get("result") or {}
    if not smoke and (design["smoke"] or report.get("smoke", True)
            or report["status"] != "COMPLETE" or not report.get("slumbot_eligible")
            or not result.get("confirmatory") or result.get("conclusion") != "improved"
            or result["interval"][0] <= 0):
        raise ValueError("Slumbot requires a completed positive local primary; no live calls made")
    seed = design["external_fit_seed"]  # Never select the best-looking fit seed.
    return design, {arm: specs[f"{arm}_{seed}"] for arm in ARMS}, {arm: hashes[f"{arm}_{seed}"] for arm in ARMS}


def run_benchmark(experiment, evaluation, output, *, max_seconds=86400, seed=571901,
                  smoke=False, client_factory=None):
    if not math.isfinite(max_seconds) or max_seconds <= 0 or seed < 0:
        raise ValueError("Invalid benchmark limits")
    if smoke and client_factory is None:
        raise ValueError("Smoke requires an offline client; no live API calls")
    design, specs, hashes = verified_gate(experiment, evaluation, smoke=smoke)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    pilot_count, main_count, block_size = (4, 8, 2) if smoke else (1000, 20000, 100)
    plan = dict(fit_seed=design["external_fit_seed"], seed=seed, alpha=.05,
        pilot_hands_per_arm=pilot_count, confirmation_hands_per_arm=main_count,
        pilot_excluded=True, fixed_sample=True, block_size=block_size,
        max_seconds=max_seconds, policy_sha256=hashes,
        experiment_sha256=sha256_file(Path(experiment) / "experiment.json"),
        local_report_sha256=sha256_file(Path(evaluation) / "report.json"),
        primary_metric="raw_bb", interval="independent-arm Welch", smoke=smoke,
        interpretation="Fresh independent hands for both arms; opponent version may be unknown; no paired-deal claim")
    write_json(output / "design.json", plan)
    write_json(output / "plan.json", plan)
    report = dict(status="STARTING", provider="slumbot", smoke=smoke, fit_seed=design["external_fit_seed"],
        primary_metric="raw_bb", policy_sha256=hashes, opponent_version_verified=False,
        scores={a: {"pilot": [], "main": []} for a in ARMS}, improvement=None)
    journal = Journal(output / "hands.jsonl")
    deadline = time.monotonic() + max_seconds
    try:
        import torch
        torch.set_num_threads(1)
        agents = {a: Agent(**specs[a], seed=seed + i) for i, a in enumerate(ARMS)}
        if agents[ARMS[0]].action_menu != agents[ARMS[1]].action_menu:
            raise ValueError("Action menus differ")
        client = (client_factory or client_for)("slumbot", agents[ARMS[0]].action_menu)
        if client.provider != "slumbot":
            raise ValueError("This benchmark accepts only Slumbot")
        before = version_for(client)
        report["opponent_version_before"] = before
        for a in ARMS:
            journal.write("policy", label=a, sha256=hashes[a])
        schedule = np.random.default_rng(seed + 100)
        for phase_id, (phase, count) in enumerate((("pilot", pilot_count), ("main", main_count))):
            report["status"] = phase.upper()
            for i, a in enumerate(ARMS):
                agents[a].rng = np.random.default_rng(seed + phase_id * 10 + i)
            for start in range(0, count, block_size):
                for a in schedule.permutation(ARMS):
                    for ordinal in range(start, min(count, start + block_size)):
                        if time.monotonic() >= deadline:
                            raise TimeoutError("Slumbot wall budget exhausted")
                        score, raw = collect_hand(client, agents[a], journal, phase, str(a), ordinal)
                        if not math.isfinite(raw) or score != raw:
                            raise ProtocolError("Expected a finite raw Slumbot score")
                        report["scores"][a][phase].append(raw)
                    write_json(output / "report.json", report)
            if phase == "pilot":
                variance = sum(float(np.var(report["scores"][a]["pilot"], ddof=1)) for a in ARMS)
                plan["estimated_detectable_effect_bb100"] = 100 * (norm.ppf(.975) + norm.ppf(.8)) * math.sqrt(variance / main_count)
                write_json(output / "plan.json", plan)
        after = version_for(client)
        changed = before is not None and after is not None and before != after
        report.update(status="INVALID_OPPONENT_VERSION_CHANGE" if changed else "COMPLETE",
            opponent_version_after=after, opponent_version_verified=before is not None and before == after)
    except TimeoutError:
        report["status"] = "INCOMPLETE_TIME_LIMIT"
    except APIError as exc:
        # Never retry an ambiguous POST or record it as a zero-score hand.
        report.update(status="INCOMPLETE_PROVIDER_ERROR", error_type=type(exc).__name__, reason=str(exc))
    except (ProtocolError, ValueError, RuntimeError) as exc:
        report.update(status="EVALUATION_ERROR", error_type=type(exc).__name__)
    finally:
        main = {a: report["scores"][a]["main"] for a in ARMS}
        complete = report["status"] == "COMPLETE" and all(len(x) == main_count for x in main.values())
        if min(map(len, main.values())) >= 2:
            report["improvement"] = estimate(main["objective_mass"], other=main["uniform"],
                complete=complete and not smoke and report["opponent_version_verified"])
            if not report["improvement"]["confirmatory"]:
                report["improvement"]["conclusion"] = ("smoke_only" if smoke else
                    "descriptive_unknown_opponent_version" if complete else "incomplete_descriptive")
        report["estimates"] = {a: estimate(v, complete=False) if len(v) >= 2 else None for a, v in main.items()}
        report["completed_hands_per_arm"] = {a: len(v) for a, v in main.items()}
        journal.close()
        write_json(output / "report.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", type=Path, required=True)
    p.add_argument("--evaluation", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-seconds", type=float, default=86400)
    p.add_argument("--seed", type=int, default=571901)
    result = run_benchmark(**vars(p.parse_args()))
    if result["status"] != "COMPLETE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
