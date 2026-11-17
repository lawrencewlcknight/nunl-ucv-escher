"""Primary duplicate-deal evaluation of matched frozen policy refits."""
import argparse
import json
import math
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np

from .checkpointing import sha256_file
from .evaluate import duplicate_pair
from .evaluation_stats import estimate, sample_plan
from .game import load_hunl_game
from .play import Agent
from .train import write_json

_agents = None
_game = None
_error = None


def verified_refits(experiment):
    root = Path(experiment).resolve()
    design = json.loads((root / "experiment.json").read_text())
    if design["status"] != "COMPLETE":
        raise ValueError("Both refit arms must finish before evaluation")
    seeds = design["fit_seeds"]
    if not seeds or len(set(seeds)) != len(seeds) or design["external_fit_seed"] != seeds[0]:
        raise ValueError("Invalid predeclared fit seeds")
    specs, hashes = {}, {}
    for seed in seeds:
        initial_hashes = []
        for arm in ("uniform", "objective_mass"):
            rows = [r for r in design["results"] if r["arm"] == arm and r["fit_seed"] == seed]
            if len(rows) != 1:
                raise ValueError("Incomplete or duplicate matched refit")
            row = rows[0]
            path = (root / row["checkpoint"]).resolve()
            if not path.is_relative_to(root) or sha256_file(path) != row["sha256"]:
                raise ValueError("Refit checkpoint path or checksum mismatch")
            name = f"{arm}_{seed}"
            specs[name] = dict(kind="checkpoint", checkpoint=str(path))
            hashes[name] = row["sha256"]
            initial_hashes.append(row["initial_model_sha256"])
        if len(set(initial_hashes)) != 1:
            raise ValueError("A/B networks did not start from identical weights")
    return design, specs, hashes


def initialise(specs):
    global _agents, _game, _error
    import torch
    torch.set_num_threads(1)
    _error = None
    try:
        _agents = {k: Agent(**v) for k, v in specs.items()}
        menu = next(iter(_agents.values())).action_menu
        if any(a.action_menu != menu for a in _agents.values()):
            raise ValueError("Refit action menus differ")
        _game = load_hunl_game(menu)
    except Exception as exc:
        _error = f"{type(exc).__name__}: {exc}"


def batch(task):
    if _error:
        raise RuntimeError(_error)
    deal_seed, start, count, fit_seeds = task
    rows = []
    for pair_id in range(start, start + count):
        scores = {}
        for seed in fit_seeds:
            agents = [_agents[f"objective_mass_{seed}"], _agents[f"uniform_{seed}"]]
            # All matched refits share the same predealt cards; aggregate within
            # each deal group before computing an interval, avoiding pseudoreplication.
            scores[str(seed)] = duplicate_pair(_game, agents, deal_seed, pair_id)
        rows.append(dict(pair_id=pair_id, legs_by_fit_seed=scores,
                         mean_bb=float(np.mean(list(scores.values())))))
    return rows


def collect(pool, *, count, seed, fit_seeds, path, deadline, workers, rows=None):
    rows = [] if rows is None else rows
    with Path(path).open("x") as handle:
        for start in range(0, count, workers * 16):
            if time.monotonic() >= deadline:
                break
            stop = min(count, start + workers * 16)
            tasks = [(seed, i, min(4, stop - i), fit_seeds) for i in range(start, stop, 4)]
            pending = pool.imap(batch, tasks)
            for _ in tasks:
                try:
                    group = pending.next(timeout=max(.001, deadline - time.monotonic()))
                except mp.TimeoutError:
                    return rows
                for row in group:
                    handle.write(json.dumps(row) + "\n")
                    rows.append(row)
                handle.flush()
    return rows


def run_evaluation(experiment, output, *, workers=8, pilot_pairs=10000,
                   minimum=100000, maximum=1000000, max_seconds=7200,
                   effect_bb100=10., seed=471901):
    if (workers < 1 or pilot_pairs < 2 or not 2 <= minimum <= maximum or seed < 0
            or not all(math.isfinite(x) and x > 0 for x in (max_seconds, effect_bb100))):
        raise ValueError("Invalid statistical or execution limits")
    root, output = Path(experiment), Path(output)
    design, specs, hashes = verified_refits(root)
    seeds = design["fit_seeds"]
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status="STARTING", smoke=design["smoke"], result=None, completed_groups=0,
                  fit_seeds=seeds, slumbot_eligible=False,
                  inference_scope="Fixed refits of one training replay; evaluation uncertainty only")
    manifest = dict(experiment_sha256=sha256_file(root / "experiment.json"), policies=hashes,
        fit_seeds=seeds, external_fit_seed=design["external_fit_seed"], pilot_seed=seed,
        confirmation_seed=seed + 1, pilot_pairs=pilot_pairs, minimum=minimum, maximum=maximum,
        max_seconds=max_seconds, effect_bb100=effect_bb100, alpha=.05, workers=workers,
        unit="shared deal group averaged over both seats and all fit seeds", pilot_excluded=True)
    write_json(output / "manifest.json", manifest)
    deadline, main = time.monotonic() + max_seconds, []
    try:
        with mp.get_context("spawn").Pool(workers, initializer=initialise, initargs=(specs,)) as pool:
            tick = time.monotonic()
            pilot = collect(pool, count=pilot_pairs, seed=seed, fit_seeds=seeds,
                path=output / "pilot.jsonl", deadline=deadline, workers=workers)
            if len(pilot) != pilot_pairs:
                report["status"] = "INCOMPLETE_PILOT"
            else:
                plan = sample_plan([r["mean_bb"] for r in pilot], minimum=minimum, maximum=maximum,
                                   effect_bb100=effect_bb100, alpha=.05)
                rate = pilot_pairs / (time.monotonic() - tick)
                affordable = int(.8 * rate * max(0, deadline - time.monotonic()))
                selected = min(plan["selected"], affordable)
                plan.update(selected=selected, capped=selected < plan["required"], affordable=affordable,
                    pilot_groups_per_second=rate,
                    estimated_detectable_effect_bb100=(plan["estimated_detectable_effect_bb100"] *
                        math.sqrt(min(maximum, max(minimum, plan["required"])) / selected) if selected > 0 else None))
                report["plan"] = plan
                write_json(output / "plan.json", plan)
                if selected < minimum:
                    report["status"] = "INSUFFICIENT_TIME_FOR_MINIMUM"
                else:
                    collect(pool, count=selected, seed=seed + 1, fit_seeds=seeds,
                        path=output / "confirmatory.jsonl", deadline=deadline, workers=workers, rows=main)
                    report["status"] = "COMPLETE" if len(main) == selected else "INCOMPLETE_TIME_LIMIT"
    except Exception as exc:
        report.update(status="EVALUATION_ERROR", error=f"{type(exc).__name__}: {exc}")
    finally:
        if len(main) >= 2:
            report["result"] = estimate([r["mean_bb"] for r in main],
                complete=report["status"] == "COMPLETE" and not design["smoke"])
            report["by_fit_seed_descriptive"] = {str(s): estimate(
                [np.mean(r["legs_by_fit_seed"][str(s)]) for r in main], complete=False) for s in seeds}
        report["slumbot_eligible"] = bool(report["result"] and report["result"]["confirmatory"]
                                         and report["result"]["conclusion"] == "improved")
        report.update(completed_groups=len(main), completed_hands=2 * len(seeds) * len(main))
        write_json(output / "report.json", report)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--pilot-pairs", type=int, default=10000)
    p.add_argument("--minimum", type=int, default=100000)
    p.add_argument("--maximum", type=int, default=1000000)
    p.add_argument("--max-seconds", type=float, default=7200)
    p.add_argument("--effect-bb100", type=float, default=10.)
    p.add_argument("--seed", type=int, default=471901)
    report = run_evaluation(**vars(p.parse_args(argv)))
    print(json.dumps(report, indent=2))
    if report["status"] != "COMPLETE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
