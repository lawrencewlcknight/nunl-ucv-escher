"""Duplicate-deal evaluation of frozen policies, isolated from training RNGs."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np

from .checkpointing import sha256_file
from .evaluation_stats import estimate, sample_plan
from .game import load_hunl_game
from .play import Agent
from .train import write_json

_agents = None
_game = None


def initialise(a, b):
    global _agents, _game
    import torch
    torch.set_num_threads(1)
    _agents = [Agent(**spec) for spec in (a, b)]
    if _agents[0].action_menu != _agents[1].action_menu:
        raise ValueError("Evaluation requires identical action menus")
    _game = load_hunl_game(_agents[0].action_menu)


def duplicate_pair(game, agents, seed, pair_id):
    """Replay the same ordered cards by SEAT, swapping agents, not cards.

    Deal randomness and all four action streams are independent. A different
    betting path cannot change the predealt board. Returns A's profit in bb.
    """
    streams = np.random.SeedSequence([seed, pair_id]).spawn(5)
    deck = np.random.default_rng(streams[0]).permutation(52)[:9]
    returns = []
    for seat_a in (0, 1):
        state, dealt = game.new_initial_state(), 0
        for i, agent in enumerate(agents):
            agent.rng = np.random.default_rng(streams[1 + 2*seat_a + i])
        for _ in range(game.max_game_length() + 16):
            if state.is_terminal():
                break
            if state.is_chance_node():
                state.apply_action(int(deck[dealt]))
                dealt += 1
            else:
                agent = agents[0 if state.current_player() == seat_a else 1]
                state.apply_action(agent.choose(state.decision_view()))
        if not state.is_terminal():
            raise RuntimeError("Evaluation hand exceeded engine bound")
        profit = state.returns()
        if abs(sum(profit)) > 1e-8:
            raise RuntimeError("Non-zero-sum settlement")
        returns.append(float(profit[seat_a]) / 100)
    return returns


def batch(task):
    seed, start, count = task
    return [duplicate_pair(_game, _agents, seed, i) for i in range(start, start+count)]


def collect(executor, *, count, seed, path, deadline):
    result = []
    # Bounded dispatch avoids queueing a million pairs or a full replay copy.
    with path.open("x") as handle:
        for start in range(0, count, 1024):
            if time.monotonic() >= deadline:
                break
            tasks = [(seed, i, min(64, count-i, start+1024-i))
                     for i in range(start, min(count, start+1024), 64)]
            groups = map(batch, tasks) if executor is None else executor.map(batch, tasks)
            for group in groups:
                for pair in group:
                    handle.write(json.dumps(pair) + "\n")
                    result.append(sum(pair)/2)
            handle.flush()
    return result


def run_evaluation(a, b, output, *, seed=8301, workers=8, pilot_pairs=10000,
                   minimum=100000, maximum=1000000, max_seconds=7200,
                   effect_bb100=10., alpha=.05):
    if workers < 1 or pilot_pairs < 2 or max_seconds <= 0:
        raise ValueError("Invalid evaluation limits")
    if not (2 <= minimum <= maximum) or not np.isfinite(max_seconds) or seed < 0:
        raise ValueError("Invalid sample, time or seed limits")
    if not (0 < alpha < 1 and np.isfinite(effect_bb100) and effect_bb100 > 0):
        raise ValueError("Invalid alpha or effect size")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    b = str(b)
    specs = [dict(kind="checkpoint", checkpoint=str(a)),
             dict(kind=b) if b in ("random", "check_call") else
             dict(kind="checkpoint", checkpoint=str(b))]
    identity = [dict(**spec, sha256=sha256_file(spec["checkpoint"]) if "checkpoint" in spec else None)
                for spec in specs]
    write_json(output / "manifest.json", dict(agents=identity, seed=seed,
               unit="independent_duplicate_pair_average_bb", pilot_pairs=pilot_pairs,
               minimum=minimum, maximum=maximum, max_seconds=max_seconds,
               effect_bb100=effect_bb100, alpha=alpha, workers=workers))
    deadline = time.monotonic() + max_seconds
    pool = None
    try:
        if workers > 1:
            pool = ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"),
                                       initializer=initialise, initargs=specs)
        else:
            initialise(*specs)
        pilot_started = time.monotonic()
        pilot = collect(pool, count=pilot_pairs, seed=seed, path=output / "pilot.jsonl", deadline=deadline)
        if len(pilot) != pilot_pairs:
            report = dict(status="INCOMPLETE_PILOT", completed_pairs=len(pilot))
        else:
            rate = pilot_pairs / (time.monotonic()-pilot_started)
            affordable = max(minimum, int(.8 * rate * max(0, deadline-time.monotonic())))
            effective_maximum = min(maximum, affordable)
            plan = sample_plan(pilot, minimum=minimum, maximum=effective_maximum, effect_bb100=effect_bb100, alpha=alpha)
            plan.update(pilot_pairs_per_second=rate, requested_maximum=maximum,
                        runtime_planning_maximum=effective_maximum)
            write_json(output / "plan.json", plan)  # Freeze BEFORE confirmatory sampling.
            main = collect(pool, count=plan["selected"], seed=seed+1,
                           path=output / "confirmatory.jsonl", deadline=deadline)
            complete = len(main) == plan["selected"]
            report = dict(status="COMPLETE" if complete else "INCOMPLETE_TIME_LIMIT",
                          plan=plan, completed_pairs=len(main),
                          result=estimate(main, alpha=alpha, complete=complete) if len(main) >= 2 else None)
        write_json(output / "report.json", report)
        return report
    finally:
        if pool is not None:
            pool.shutdown()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--a", required=True, type=Path, help="Later checkpoint")
    p.add_argument("--b", required=True, help="Earlier checkpoint, random, or check_call")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=8301)
    p.add_argument("--pilot-pairs", type=int, default=10000)
    p.add_argument("--minimum", type=int, default=100000)
    p.add_argument("--maximum", type=int, default=1000000)
    p.add_argument("--max-seconds", type=float, default=7200)
    p.add_argument("--alpha", type=float, default=.05)
    args = p.parse_args(argv)
    print(json.dumps(run_evaluation(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
