"""Small inference-only runner: offline verification or bounded API matches."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

import numpy as np

from .api import APIError, GTOWizardClient, GTOWizardTransport, JSONTransport, SlumbotClient
from .checkpointing import LoadedHUNLPolicy, load_checkpoint_payload, sha256_file
from .features import ENCODER_ID
from .game import DEFAULT_ACTION_MENU, load_hunl_game
from .offline import run_equivalence
from .remote import ProtocolError


class Agent:
    def __init__(self, kind="check_call", *, seed=0, checkpoint=None):
        self.kind, self.rng = kind, np.random.default_rng(seed)
        self.policy = None
        self.action_menu = DEFAULT_ACTION_MENU
        self.checkpoint_hash = None
        if kind == "checkpoint":
            if checkpoint is None:
                raise ValueError("Checkpoint agent requires --checkpoint")
            payload = load_checkpoint_payload(checkpoint)
            self.action_menu = tuple(payload["game"]["action_menu"])
            self.policy = LoadedHUNLPolicy(load_hunl_game(self.action_menu), checkpoint)
            self.checkpoint_hash = sha256_file(checkpoint)
        elif kind not in ("check_call", "random", "all_in"):
            raise ValueError("Unknown agent")
        elif checkpoint is not None:
            raise ValueError("Select --agent checkpoint when supplying a checkpoint")

    def choose(self, view):
        legal = np.flatnonzero(view.legal_mask)
        if self.policy is not None:
            probabilities = self.policy.action_probabilities_from_view(view)
            indices = list(probabilities)
            weights = np.asarray([probabilities[a] for a in indices], dtype=np.float64)
            return int(self.rng.choice(indices, p=weights / weights.sum()))
        if self.kind == "check_call":
            return 1
        if self.kind == "all_in":
            index = self.action_menu.index("all_in")
            return index if view.legal_mask[index] else 1
        return int(self.rng.choice(legal))


class Journal:
    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open("x", encoding="utf-8")

    def write(self, event, **data):
        self.file.write(json.dumps(dict(event=event, time_unix=time.time(), **data), allow_nan=False) + "\n")
        self.file.flush()
        os.fsync(self.file.fileno())

    def close(self):
        self.file.close()


def run_matches(client, agent, *, hands, journal, resume_hand=None):
    if hands < 1 or (resume_hand is not None and (hands != 1 or client.provider != "gtow")):
        raise ValueError("Resume requires exactly one GTO Wizard hand")
    results = []
    try:
        for hand_index in range(hands):
            journal.write("request_new_hand" if resume_hand is None else "request_resume", hand_index=hand_index, hand_id=resume_hand)
            state = client.new_hand() if resume_hand is None else client.resume_hand(resume_hand)
            journal.write("observation", **state.log_record())
            decisions = 0
            while not state.terminal:
                decisions += 1
                if decisions > 218:
                    raise ProtocolError("Hand exceeded supported decision bound")
                index = agent.choose(state.view)
                kind, amount = state.action(index)
                journal.write("action_intent", hand_id=state.hand_id, menu_index=index,
                              action=kind, street_raise_to=amount,
                              observation=asdict(state.view))
                state = client.act(index)
                journal.write("observation", **state.log_record())
            results.append(state)
            journal.write("hand_complete", **state.log_record())
        total = sum(s.winnings for s in results)
        summary = dict(provider=client.provider, completed_hands=len(results), winnings_chips=total,
                       raw_bb_per_100=total / len(results),
                       recoveries=getattr(client, "recoveries", 0),
                       purpose="interface_smoke_not_strength_estimate")
        journal.write("summary", **summary)
        return summary
    except (APIError, ProtocolError, ValueError) as exc:
        journal.write("stopped", error_type=type(exc).__name__, reason=str(exc), completed_hands=len(results))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opponent", choices=("offline", "slumbot", "gtow"), default="offline")
    parser.add_argument("--hands", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--agent", choices=("check_call", "random", "all_in", "checkpoint"), default="check_call")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="New JSONL journal; existing files are never overwritten")
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--min-interval", type=float, default=.5)
    parser.add_argument("--gto-key-env", default="GTO_WIZARD_API_KEY")
    parser.add_argument("--slumbot-login", action="store_true", help="Opt into account play using SLUMBOT_USERNAME/PASSWORD; default is anonymous")
    parser.add_argument("--resume-hand", type=int)
    args = parser.parse_args(argv)
    if args.hands < 1 or args.timeout <= 0 or args.min_interval < .5:
        parser.error("Use positive hands/timeout and at least 0.5 seconds between requests")
    if args.resume_hand is not None and (args.opponent != "gtow" or args.hands != 1):
        parser.error("--resume-hand requires --opponent gtow --hands 1")
    import torch
    torch.set_num_threads(1)
    agent = Agent(args.agent, seed=args.seed, checkpoint=args.checkpoint)
    key = os.environ.get(args.gto_key_env) if args.opponent == "gtow" else None
    if args.opponent == "gtow" and not key:
        parser.error(f"Set {args.gto_key_env} to an approved researcher API key")
    if args.slumbot_login and (args.opponent != "slumbot" or not os.environ.get("SLUMBOT_USERNAME") or not os.environ.get("SLUMBOT_PASSWORD")):
        parser.error("--slumbot-login requires Slumbot and both credential environment variables")
    journal = Journal(args.output)
    try:
        journal.write("run", opponent=args.opponent, agent=args.agent, seed=args.seed,
                      requested_hands=args.hands, checkpoint_sha256=agent.checkpoint_hash,
                      action_menu=list(agent.action_menu), encoder_id=ENCODER_ID)
        if args.opponent == "offline":
            summary = run_equivalence(args.hands, seed=args.seed, action_menu=agent.action_menu, policy=agent.policy)
            journal.write("summary", **summary)
        else:
            base = "https://slumbot.com" if args.opponent == "slumbot" else "https://researcher.gtowizard.com"
            transport_type = GTOWizardTransport if args.opponent == "gtow" else JSONTransport
            transport = transport_type(base, headers={"X-API-Key": key} if key else None,
                                      timeout=args.timeout, min_interval=args.min_interval)
            if args.opponent == "slumbot":
                client = SlumbotClient(action_menu=agent.action_menu, transport=transport,
                    username=os.environ.get("SLUMBOT_USERNAME") if args.slumbot_login else None,
                    password=os.environ.get("SLUMBOT_PASSWORD") if args.slumbot_login else None)
            else:
                client = GTOWizardClient(action_menu=agent.action_menu, transport=transport)
            summary = run_matches(client, agent, hands=args.hands, journal=journal, resume_hand=args.resume_hand)
        print(json.dumps(summary, indent=2))
    finally:
        journal.close()


if __name__ == "__main__":
    main()
