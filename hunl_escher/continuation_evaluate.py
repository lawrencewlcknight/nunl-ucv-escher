"""Five fixed-policy comparisons after a verified 24-hour continuation."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

from .checkpointing import sha256_file
from .continuation import LABELS
from .experiment_runtime import run_child
from .train import write_json

# Conservative family correction: four secondary tests at 1% each.
SECONDARY_ALPHA = .01


def verified_checkpoint(root, label, *, smoke=False):
    root = Path(root)
    training = json.loads((root / 'continuation.json').read_text())
    if training['status'] != 'TRAINING_COMPLETE' or not training.get('final_recovery_verified'):
        raise ValueError('Evaluate only after all stages and the final reload check complete')
    if bool(training['smoke']) != bool(smoke):
        raise ValueError('Explicit smoke mode must match training')
    marker = json.loads((root / f'{label}.json').read_text())
    saved = training['checkpoints'][label]
    if any(marker[key] != saved[key] for key in ('sha256', 'outer_iteration', 'active_seconds')):
        raise ValueError('Policy marker differs from completed training record')
    policy = root / f'{label}.pkl'
    if sha256_file(policy) != marker['sha256']:
        raise ValueError('Policy hash differs from completed training record')
    return policy, marker


def gate_secondary(result, primary_passed, smoke):
    if result and (not primary_passed or smoke):
        result.update(confirmatory=False,
                      conclusion='smoke_only' if smoke else 'descriptive_primary_gate_not_passed')
    return result


def render(output, report):
    lines = ['# No-limit Holdem continuation: local evaluation', '',
             'Primary: P32 vs P8, 95% interval. Four adjacent comparisons: 99% intervals;',
             'secondary improvement claims require the primary test to pass.',
             'One training seed; these matches do not measure exploitability.', '',
             '| Comparison | Status | bb/100 | Interval | Conclusion |',
             '|---|---|---:|---|---|']
    for name, item in report['head_to_head'].items():
        result = item.get('result')
        if result:
            lo, hi = result['interval']
            lines.append(f"| {name} | {item['status']} | {result['bb_per_100']:.2f} | "
                         f"{100*result['confidence']:.0f}% [{lo:.2f}, {hi:.2f}] | {result['conclusion']} |")
        else:
            lines.append(f"| {name} | {item['status']} | — | — | — |")
    lines += ['', 'Slumbot is a separate final-policy-only job; see benchmarks/slumbot/report.json.']
    (Path(output) / 'continuation_report.md').write_text('\n'.join(lines) + '\n')


def run_suite(experiment, output, *, workers=8, max_seconds=14400, smoke=False):
    if workers < 1 or not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError('Invalid evaluation limits')
    root, output = Path(experiment).resolve(), Path(output).resolve()
    for label in LABELS:
        verified_checkpoint(root, label, smoke=smoke)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status='EVALUATING', smoke=smoke, primary='P32-vs-P8',
                  secondary_alpha=SECONDARY_ALPHA, secondary_family_size=4, head_to_head={})
    write_json(output / 'design.json', dict(labels=LABELS, primary=report['primary'],
        primary_alpha=.05, secondary_alpha=SECONDARY_ALPHA, secondary_family_size=4,
        effect_bb100=10, power=.8, pilot_excluded=True, workers=workers,
        max_seconds=max_seconds, smoke=smoke,
        pilot_pairs=4 if smoke else 10000, minimum=8 if smoke else 100000,
        maximum=8 if smoke else 1000000))
    deadline = time.monotonic() + max_seconds
    primary_passed = False
    matches = [('P32', 'P8'), *zip(LABELS[1:], LABELS[:-1])]
    for index, (a, b) in enumerate(matches):
        name = a + '-vs-' + b
        remaining = deadline - time.monotonic()
        budget = remaining * .4 if index == 0 else remaining / (len(matches) - index)
        if budget <= 1:
            item = dict(status='SKIPPED_TIME_LIMIT', result=None)
        else:
            target = output / name
            command = [sys.executable, '-m', 'hunl_escher.evaluate',
                '--a', str(root / f'{a}.pkl'), '--b', str(root / f'{b}.pkl'),
                '--output', str(target), '--workers', str(workers),
                '--max-seconds', str(budget), '--seed', str(28301 + 10*index),
                '--alpha', str(.05 if index == 0 else SECONDARY_ALPHA)]
            if smoke:
                command += ['--pilot-pairs', '4', '--minimum', '8', '--maximum', '8']
            # A process boundary enforces the deadline even during a stalled pool.
            outcome = run_child(command, output / f'{name}.log', timeout=budget)
            path = target / 'report.json'
            item = json.loads(path.read_text()) if path.exists() else dict(result=None)
            if outcome['returncode'] != 0:
                item['status'] = 'INCOMPLETE_TIME_LIMIT' if outcome['timed_out'] else 'EVALUATION_ERROR'
                if item.get('result'):
                    item['result'].update(confirmatory=False, conclusion='incomplete')
        if index == 0:
            primary_passed = (not smoke and item['status'] == 'COMPLETE' and
                (item.get('result') or {}).get('confirmatory', False) and
                item['result']['conclusion'] == 'improved')
            if smoke:
                gate_secondary(item.get('result'), False, True)
        else:
            gate_secondary(item.get('result'), primary_passed, smoke)
        report['head_to_head'][name] = item
        if (output / name).exists():
            write_json(output / name / 'report.json', item)
        write_json(output / 'report.json', report)
    report.update(primary_passed=primary_passed,
                  status='COMPLETE' if all(x['status'] == 'COMPLETE' for x in report['head_to_head'].values())
                  else 'EVALUATION_INCOMPLETE')
    write_json(output / 'report.json', report)
    render(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--max-seconds', type=float, default=14400)
    parser.add_argument('--smoke', action='store_true')
    if run_suite(**vars(parser.parse_args()))['status'] != 'COMPLETE':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
