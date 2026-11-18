"""Bounded, resumable audit with frozen provenance and human-readable reports."""
from dataclasses import replace, asdict
from importlib.metadata import version
from itertools import permutations
import html
import json
import math
import os
from pathlib import Path
import platform
import time

import numpy as np

from ..checkpointing import LoadedHUNLPolicy, load_checkpoint_payload, sha256_file
from ..game import DEFAULT_ACTION_MENU, load_hunl_game
from .all_in import exact_all_in
from .engine import PANELS, RulePolicy, BudgetExpired, check_time, exact_river, probabilities
from .library import digest, state_for, validate_library, write_new


def atomic_json(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w") as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def source_identity():
    root = Path(__file__).resolve().parents[2]
    files = {}
    for name in ("hunl_escher", "vr_deep_cfr", "unbiased_escher", "adaptive_escher"):
        for p in sorted((root/name).rglob("*.py")):
            files[str(p.relative_to(root))] = sha256_file(p)
    return digest(files)


def policy_spec(label, value):
    if not label:
        raise ValueError("Policy labels cannot be empty")
    if value.startswith("rule:"):
        kind = value[5:]
        RulePolicy(kind)
        return dict(label=label, kind=kind, action_menu=list(DEFAULT_ACTION_MENU))
    p = Path(value).resolve()
    # Only local, trusted playable policy checkpoints, never full training states.
    payload = load_checkpoint_payload(p)
    return dict(label=label, kind="checkpoint", path=str(p), sha256=sha256_file(p),
        action_menu=payload["game"]["action_menu"], encoder=payload.get("feature_encoder"),
        iteration=payload.get("outer_iteration"), training_seconds=payload.get("training_elapsed_seconds"),
        training_seed=payload.get("seed"))


def load_policy(spec):
    if spec["kind"] != "checkpoint":
        return RulePolicy(spec["kind"])
    if sha256_file(spec["path"]) != spec["sha256"]:
        raise ValueError("Checkpoint changed after manifest was frozen")
    return LoadedHUNLPolicy(load_hunl_game(spec["action_menu"]), spec["path"])


def observation_checks(position, state, policy, *, suits=False, deadline=float("inf")):
    view = state.decision_view()
    p = probabilities(policy, view)
    available = [c for c in range(52) if c not in (*view.hole_cards, *view.board_cards)]
    alternative = state_for(position, view.action_menu, tuple(available[-2:])).decision_view()
    if view != alternative:
        raise RuntimeError("Hidden placeholder cards changed the public decision view")
    swapped = probabilities(policy, replace(view, hole_cards=tuple(reversed(view.hole_cards))))
    flop_order = view.board_cards
    if len(flop_order) >= 3:
        flop_order = (*reversed(flop_order[:3]), *flop_order[3:])
    flop = probabilities(policy, replace(view, board_cards=flop_order))
    suit_tv = None
    if suits:
        suit_tv = 0.
        for perm in permutations(range(4)):
            check_time(deadline)
            def remap(cs):
                return tuple(4*(c//4)+perm[c%4] for c in cs)
            q = probabilities(policy, replace(view, hole_cards=remap(view.hole_cards),
                                               board_cards=remap(view.board_cards)))
            suit_tv = max(suit_tv, float(np.abs(p-q).sum()/2))
    return dict(hidden_card_view_identical=True,
                hole_order_total_variation=float(np.abs(p-swapped).sum()/2),
                flop_order_total_variation=float(np.abs(p-flop).sum()/2),
                max_suit_total_variation=suit_tv)


def audit_position(position, policy, menu, *, suits, deadline):
    check_time(deadline)
    state = state_for(position, menu)
    view = state.decision_view()
    p = probabilities(policy, view)
    checks = observation_checks(position, state, policy, suits=suits, deadline=deadline)
    row = dict(position_id=position["id"], family=position["family"], tags=position["tags"],
        position=position, view=asdict(view),
        probabilities={name: float(p[i]) for i, name in enumerate(menu) if view.legal_mask[i]},
        entropy_nats=float(-sum(x*math.log(x) for x in p if x > 0)),
        aggression_probability=float(sum(p[i] for i in range(2, len(p)))),
        checks=checks, river=[], all_in=None)
    if view.street == 3:
        row["river"] = [exact_river(state, policy, panel, deadline=deadline) for panel in PANELS]
    if view.street in (1, 2) and set(state.legal_actions()) == {0, 1}:
        row['all_in'] = exact_all_in(state, policy, deadline=deadline)
    check_time(deadline)
    return row


def balanced(rows, metric):
    groups = {}
    for r in rows:
        groups.setdefault(r["family"], []).append(float(r[metric]))
    return float(np.mean([np.mean(v) for v in groups.values()])) if groups else None


def summarize(rows_by_policy, *, complete):
    summary, comparisons = {}, []
    for label, rows in rows_by_policy.items():
        panels = {}
        for panel in PANELS:
            values = [dict(family=r["family"], **s) for r in rows for s in r["river"] if s["panel"] == panel]
            panels[panel] = dict(positions=len(values), board_families=len({r["family"] for r in values}),
                balanced_gap_bb=balanced(values, "local_gap_bb"),
                balanced_gap_pot_fraction=balanced(values, "gap_pot_fraction"),
                balanced_deployed_value_bb=balanced(values, "deployed_value_bb"),
                balanced_universal_fold_error_mass=balanced(values, "universal_fold_error_mass"),
                balanced_universal_call_error_mass=balanced(values, "universal_call_error_mass"),
                balanced_guaranteed_cost_bb=balanced(values, "guaranteed_cost_bb"))
        by_tag = {}
        for tag in sorted({t for r in rows for t in r['tags']}):
            selected = [r for r in rows if tag in r['tags']]
            by_tag[tag] = dict(positions=len(selected), mean_aggression_probability=float(np.mean(
                [r['aggression_probability'] for r in selected])),
                panels={panel: balanced([dict(family=r['family'], **s) for r in selected
                    for s in r['river'] if s['panel']==panel], 'local_gap_bb') for panel in PANELS})
        checks = [r["checks"] for r in rows]
        all_in_rows = [dict(family=r['family'], **r['all_in']) for r in rows if r.get('all_in')]
        summary[label] = dict(completed_positions=len(rows), panels=panels, by_tag=by_tag,
            all_in=dict(positions=len(all_in_rows), balanced_gap_bb=balanced(all_in_rows,'local_gap_bb')),
            maximum_hole_order_tv=max((c["hole_order_total_variation"] for c in checks), default=None),
            maximum_flop_order_tv=max((c["flop_order_total_variation"] for c in checks), default=None),
            maximum_suit_tv=max((c["max_suit_total_variation"] for c in checks
                                if c["max_suit_total_variation"] is not None), default=None))
    labels = list(rows_by_policy)
    if complete:
        for i, a in enumerate(labels):
            for b in labels[i+1:]:
                by_a = {r["position_id"]: r for r in rows_by_policy[a]}
                by_b = {r["position_id"]: r for r in rows_by_policy[b]}
                for panel in (*PANELS, "uniform_all_in"):
                    deltas = []
                    for key in sorted(by_a.keys() & by_b.keys()):
                        ar, br = by_a[key], by_b[key]
                        av = (ar.get('all_in') if panel == 'uniform_all_in' else
                              next((x for x in ar["river"] if x["panel"] == panel), None))
                        bv = (br.get('all_in') if panel == 'uniform_all_in' else
                              next((x for x in br["river"] if x["panel"] == panel), None))
                        if av is not None and bv is not None:
                            deltas.append(dict(family=ar["family"], delta=bv["local_gap_bb"]-av["local_gap_bb"], value_delta=bv['deployed_value_bb']-av['deployed_value_bb']))
                    if not deltas:
                        continue
                    comparisons.append(dict(a=a, b=b, panel=panel, matched_positions=len(deltas),
                        gap_delta_b_minus_a_bb=balanced(deltas, "delta"),
                        deployed_value_delta_b_minus_a_bb=balanced(deltas, "value_delta")))
    return dict(policies=summary, comparisons=comparisons,
        interpretation="Lower gap is better against this fixed synthetic panel only. Exact conditional values; no population CI or whole-game strength claim.",
        units="bb per audited decision; equal board-family weight, equal positions within family",
        comparisons_suppressed=not complete)


def render_html(report, rows_by_policy):
    esc = lambda x: html.escape(str(x))
    parts = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>HUNL strategic audit</title>',
        '<style>body{font:16px system-ui;max-width:1200px;margin:40px auto;padding:0 20px;color:#172b3a}table{border-collapse:collapse;width:100%;margin:20px 0}td,th{border-bottom:1px solid #ccc;padding:9px;text-align:left}pre{white-space:pre-wrap}summary{cursor:pointer}small{color:#526575}</style>',
        '<h1>HUNL strategic-position audit</h1>', f'<p>Status: <b>{esc(report["status"])}</b></p>',
        '<p><b>Smoke library: execution evidence only.</b></p>' if report['smoke'] else '',
        '<p>River values integrate all 990 compatible holdings exactly. Ranges are stipulated stress tests, not inferred equilibrium ranges. Earlier streets show behaviour; fold-or-call flop/turn all-ins also receive exact runout equity under a uniform range.</p>',
        '<p>Costs are big blinds per audited decision, not bb/100 hands. These fixed positions do not establish whole-game improvement. Checking the nuts or bluffing a weak hand is not automatically an error.</p>',
        '<table><tr><th>Policy</th><th>Panel</th><th>Positions</th><th>Balanced local gap (bb)</th><th>Guaranteed cost floor (bb)</th></tr>']
    for label, s in report["summary"]["policies"].items():
        for panel, v in s["panels"].items():
            parts.append('<tr>'+''.join(f'<td>{esc(x)}</td>' for x in (label, panel, v['positions'],
                v['balanced_gap_bb'], v['balanced_guaranteed_cost_bb']))+'</tr>')
    parts.append('</table><h2>Matched checkpoint changes</h2><pre>'+esc(json.dumps(report['summary']['comparisons'], indent=2))+'</pre>')
    parts.append('<h2>Largest decision gaps</h2><p>Examples are deliberately selected extremes; read them with the balanced scores above.</p>')
    for label, rows in rows_by_policy.items():
        ordered = sorted(rows, key=lambda r:max((v['local_gap_bb'] for v in r['river']), default=-1), reverse=True)[:12]
        for r in ordered:
            parts.append(f'<details><summary>{esc(label)} · {esc(r["position_id"])}</summary><pre>{esc(json.dumps(r, indent=2))}</pre></details>')
    parts.append('<h2>Categories</h2><p>Tags overlap; their scores must not be added together.</p>')
    for label, summary in report['summary']['policies'].items():
        parts.append(f'<details><summary>{esc(label)}: category and implementation checks</summary><pre>{esc(json.dumps(summary, indent=2))}</pre></details>')
    parts.append('<h2>All positions, actions and values</h2>')
    for label, rows in rows_by_policy.items():
        parts.append(f'<details><summary>{esc(label)} ({len(rows)} positions)</summary>')
        for r in rows:
            parts.append(f'<details><summary>{esc(r["position_id"])}</summary><pre>{esc(json.dumps(r, indent=2))}</pre></details>')
        parts.append('</details>')
    parts.append('</html>')
    return '\n'.join(parts)


def run(library_path, policies, output, *, partition="development", unlock_assessment=False,
        max_seconds=300., threads=1, resume=False):
    if not np.isfinite(max_seconds) or max_seconds <= 0 or type(threads) is not int or threads < 1:
        raise ValueError("Time and thread limits must be positive")
    if partition not in ("development", "assessment"):
        raise ValueError("Invalid partition")
    if partition == "assessment" and not unlock_assessment:
        raise ValueError("Assessment requires --unlock-assessment after model selection is frozen")
    if not policies or len({p[0] for p in policies}) != len(policies):
        raise ValueError("Provide uniquely labelled policies")
    import torch
    torch.set_num_threads(threads)
    started = time.monotonic()
    deadline = started + max_seconds
    library = validate_library(json.loads(Path(library_path).read_text()))
    positions = [p for p in library["positions"] if p["partition"] == partition]
    if not positions:
        raise ValueError("No positions in selected partition")
    specs = [policy_spec(*p) for p in policies]
    if any(s['action_menu'] != specs[0]['action_menu'] for s in specs):
        raise ValueError("Comparisons require identical action menus")
    manifest = dict(schema="hunl_strategic_audit_v1", library_sha256=digest(library),
        library=library, policies=specs, partition=partition, assessment_unlocked=unlock_assessment,
        threads=threads, panels=list(PANELS), source_sha256=source_identity(),
        runtime=dict(python=platform.python_version(), numpy=version('numpy'), torch=version('torch'),
                     open_spiel=version('open_spiel')), weighting="equal family, then equal position",
        range_contract="Stipulated conditional river priors; no claim to infer betting-history reach",
        continuation="Candidate unchanged; synthetic opponent uses only fold/call/jam; exact full river continuation")
    out = Path(output)
    if resume:
        if not out.is_dir() or json.loads((out/'manifest.json').read_text()) != manifest:
            raise ValueError("Resume requires identical library, policies, source, runtime and threads")
    else:
        out.mkdir(parents=True, exist_ok=False)
        write_new(out/'manifest.json', manifest)
    lock = out/'audit.lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(fd)
    rows_by_policy = {s['label']: [] for s in specs}
    status, error = 'COMPLETE', None
    try:
        records = out/'records'
        records.mkdir(exist_ok=True)
        # Only one checkpoint model in memory at a time.
        for spec in specs:
            check_time(deadline)
            policy = load_policy(spec)
            seen_families = set()
            for position in positions:
                check_time(deadline)
                key = digest([spec['label'], position['id']])
                path = records/(key+'.json')
                if path.exists():
                    envelope = json.loads(path.read_text())
                    row = envelope['row']
                    if envelope['sha256'] != digest(row) or row['position_id'] != position['id']:
                        raise ValueError("Stored position checksum/identity mismatch")
                else:
                    row = audit_position(position, policy, tuple(spec['action_menu']),
                        suits=position['family'] not in seen_families, deadline=deadline)
                    atomic_json(path, dict(row=row, sha256=digest(row)))
                rows_by_policy[spec['label']].append(row)
                seen_families.add(position['family'])
            del policy
    except BudgetExpired as exc:
        status, error = 'INCOMPLETE_TIME_LIMIT', str(exc)
    except BaseException as exc:
        status, error = 'FAILED', f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report = dict(status=status, error=error, smoke=bool(library.get('smoke')),
            expected_positions_per_policy=len(positions), elapsed_seconds=time.monotonic()-started,
            manifest_sha256=digest(manifest), summary=summarize(rows_by_policy, complete=status=='COMPLETE'))
        try:
            atomic_json(out/'report.json', report)
            (out/'report.html').write_text(render_html(report, rows_by_policy))
        finally:
            lock.unlink()
    return report
