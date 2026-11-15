"""One isolated external benchmark job; optional providers cannot fail training."""
import argparse
import json
import os
import math
from pathlib import Path
import time

import numpy as np

from .api import APIError, GTOWizardClient, JSONTransport, SlumbotClient
from .checkpointing import sha256_file
from .evaluation_stats import estimate, sample_plan
from .play import Agent, Journal, run_matches
from .remote import ProtocolError
from .train import write_json


class TaggedJournal:
    def __init__(self, journal, provider, phase, policy, ordinal):
        self.journal = journal
        self.provider = provider
        self.tags = dict(phase=phase, policy=policy, ordinal=ordinal)
        self.result = None

    def write(self, event, **data):
        if event == "summary":
            return  # run_matches' interface-smoke summary is not a strength test.
        self.journal.write(event, **self.tags, **data)
        if event == "hand_complete":
            self.result = data


def collect_hand(client, agent, journal, phase, label, ordinal):
    tagged = TaggedJournal(journal, client.provider, phase, label, ordinal)
    run_matches(client, agent, hands=1, journal=tagged)
    raw = tagged.result["winnings_bb"]
    if client.provider == "gtow":
        adjusted = tagged.result["adjusted_scores"].get("aivat_score_chips")
        if adjusted is None:
            raise ProtocolError("Completed GTO hand lacks required AIVAT score")
        score = adjusted / 100
    else:
        score = raw
    return float(score), float(raw)


def client_for(provider, menu):
    if provider == "gtow":
        return GTOWizardClient(os.environ["GTO_WIZARD_API_KEY"], action_menu=menu)
    return SlumbotClient(action_menu=menu, transport=JSONTransport("https://slumbot.com"))


def version_for(client):
    if client.provider != "gtow":
        return None
    try:
        return client.transport.request("GET", "/results?game_name=HUNL+200BB").get("gto_wizard_version")
    except (APIError, AttributeError):
        return None


def wait_policy(root, label, deadline):
    path = root / f"{label}.json"
    while time.monotonic() < deadline:
        if path.exists():
            record = json.loads(path.read_text())
            policy = root / f"{label}.pkl"
            if policy.exists() and sha256_file(policy) == record["sha256"]:
                return policy
        if (root / "training_failed.json").exists():
            return None
        time.sleep(min(5, max(0, deadline-time.monotonic())))
    return None


def run_benchmark(provider, experiment, output, *, seed=9501, pilot_hands=1000,
                  minimum=10000, maximum=20000, max_seconds=86400,
                  block_size=100, client_factory=None):
    """Wait for P4/P8, pilot each, then interleave independent main arms.

    Timeouts/unknown terminal outcomes prohibit confirmatory claims. This job
    deliberately does not auto-restart ambiguous POSTs or overwrite a journal.
    """
    if provider not in ("slumbot", "gtow") or pilot_hands < 2 or block_size < 1 or max_seconds <= 0:
        raise ValueError("Invalid benchmark design")
    if not (2 <= minimum <= maximum) or not math.isfinite(max_seconds) or seed < 0:
        raise ValueError("Invalid statistical, time or seed limits")
    root, output = Path(experiment), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(provider=provider, status="WAITING", pilot_hands=pilot_hands,
                  primary_metric="aivat_bb" if provider == "gtow" else "raw_bb",
                  alpha=.025, purpose="secondary_checkpoint_comparison",
                  claim_scope="Secondary tests; overall improvement claims require the primary head-to-head to pass",
                  scores={label: {phase: [] for phase in ("pilot", "main")} for label in ("P4", "P8")},
                  raw={label: {phase: [] for phase in ("pilot", "main")} for label in ("P4", "P8")})
    def save():
        write_json(output / "report.json", report)
    if provider == "gtow" and client_factory is None and not os.environ.get("GTO_WIZARD_API_KEY"):
        report.update(status="SKIPPED_UNAVAILABLE", reason="No approved GTO Wizard API key")
        save()
        return report
    import torch
    torch.set_num_threads(1)
    journal = Journal(output / "hands.jsonl")
    deadline = time.monotonic() + max_seconds
    client = None
    def take(label, phase, count):
        for _ in range(count):
            if time.monotonic() >= deadline:
                raise TimeoutError("Benchmark wall-time budget exhausted")
            ordinal = len(report["scores"][label][phase])
            score, raw = collect_hand(client, agents[label], journal, phase, label, ordinal)
            report["scores"][label][phase].append(score)
            report["raw"][label][phase].append(raw)
        save()
    try:
        agents = {}
        for index, label in enumerate(("P4", "P8")):
            policy = wait_policy(root, label, deadline)
            if policy is None:
                raise TimeoutError(f"No verified {label} policy before deadline/training failure")
            agents[label] = Agent("checkpoint", checkpoint=policy, seed=seed+index)
            journal.write("policy", label=label, sha256=agents[label].checkpoint_hash)
            if client is None:
                client = (client_factory or client_for)(provider, agents[label].action_menu)
            elif agents[label].action_menu != agents["P4"].action_menu:
                raise ValueError("Checkpoint action menus differ")
            report["status"] = f"PILOT_{label}"
            # P4 pilot starts as soon as the first policy is published.
            for start in range(0, pilot_hands, block_size):
                take(label, "pilot", min(block_size, pilot_hands-start))
        plan = sample_plan(report["scores"]["P8"]["pilot"], other=report["scores"]["P4"]["pilot"],
                           minimum=minimum, maximum=maximum, alpha=.025)
        write_json(output / "plan.json", plan)
        report.update(plan=plan, status="MAIN", opponent_version_before=version_for(client))
        save()
        rng = np.random.default_rng(seed+2)
        for start in range(0, plan["selected"], block_size):
            for label in rng.permutation(["P4", "P8"]):
                take(str(label), "main", min(block_size, plan["selected"]-start))
        after = version_for(client)
        before = report["opponent_version_before"]
        changed = before is not None and after is not None and before != after
        report.update(status="INVALID_OPPONENT_VERSION_CHANGE" if changed else "COMPLETE",
                      opponent_version_after=after, opponent_version_verified=before is not None and after == before)
    except TimeoutError:
        report.update(status="INCOMPLETE_TIME_LIMIT")
    except APIError as exc:
        # APIError text is sanitized by the transport; never log server bodies/keys.
        unavailable = str(exc) in ("HTTP 401", "HTTP 403")
        report.update(status="SKIPPED_UNAVAILABLE" if unavailable else "INCOMPLETE_PROVIDER_ERROR",
                      error_type=type(exc).__name__, reason=str(exc))
    except (ProtocolError, ValueError, RuntimeError) as exc:
        report.update(status="EVALUATION_ERROR", error_type=type(exc).__name__)
    finally:
        complete = report["status"] == "COMPLETE"
        main = {label: report["scores"][label]["main"] for label in ("P4", "P8")}
        report["estimates"] = {label: estimate(x, complete=complete) if len(x) >= 2 else None
                               for label, x in main.items()}
        report["improvement"] = (estimate(main["P8"], other=main["P4"], alpha=.025, complete=complete)
                                 if min(map(len, main.values())) >= 2 else None)
        report["raw_estimates"] = {label: estimate(report["raw"][label]["main"], complete=complete)
                                   if len(main[label]) >= 2 else None for label in main}
        journal.close()
        save()
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--provider", choices=("slumbot", "gtow"), required=True)
    p.add_argument("--experiment", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-seconds", type=float, default=86400)
    p.add_argument("--pilot-hands", type=int, default=1000)
    p.add_argument("--minimum", type=int, default=10000)
    p.add_argument("--maximum", type=int, default=20000)
    p.add_argument("--seed", type=int, default=9501)
    args = p.parse_args(argv)
    result = run_benchmark(**vars(args))
    print(json.dumps({k: v for k, v in result.items() if k not in ("scores", "raw")}, indent=2))


if __name__ == "__main__":
    main()
