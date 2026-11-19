"""Cost-bounded final-policy Slumbot benchmark with an optional historical baseline."""
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
from .continuation_evaluate import verified_checkpoint
from .continuation import report_labels
from .evaluation_stats import estimate
from .play import Agent, Journal
from .remote import ProtocolError
from .train import write_json


def historical_comparison(path, root, main):
    """Historical scores are always descriptive, even when their interval excludes zero."""
    result = dict(status='UNAVAILABLE', confirmatory=False)
    if path is None:
        return result
    path, root = Path(path), Path(root)
    try:
        old = json.loads(path.read_text())
        training = json.loads((root / 'continuation.json').read_text())
        baseline_label = report_labels(training)[0]
        marker = json.loads(path.with_name(f'{baseline_label}.json').read_text())
        baseline = json.loads((root / f'{baseline_label}.json').read_text())
        if marker['sha256'] != baseline['sha256'] or marker['sha256'] != training['checkpoints'][baseline_label]['sha256']:
            raise ValueError('Historical baseline identity mismatch')
        if old['provider'] != 'slumbot' or old['primary_metric'] != 'raw_bb':
            raise ValueError('Historical report is not raw Slumbot scoring')
        if old.get('smoke', False):
            raise ValueError('Historical report is a smoke test')
        if 'policy' in old:
            if old['policy'] != baseline_label or old['sha256'] != marker['sha256']:
                raise ValueError('Historical policy identity differs')
            values = old['scores']['main']
            # Completed final-policy benchmarks use a fixed 20,000-hand main sample.
            if old['status'] != 'COMPLETE' or len(values) != 20000:
                return dict(status='BASELINE_INCOMPLETE', confirmatory=False)
        else:
            selected = old['plan']['selected']
            values = old['scores'][baseline_label]['main']
            if old['status'] != 'COMPLETE' or any(len(old['scores'][label]['main']) != selected for label in ('P4', 'P8')):
                return dict(status='BASELINE_INCOMPLETE', confirmatory=False)
        result = estimate(main, other=values, complete=False)
        result.update(status='AVAILABLE', confirmatory=False, conclusion='descriptive_historical_comparison',
            report_sha256=sha256_file(path), baseline_sha256=marker['sha256'],
            interpretation='Independent-arm Welch interval; different evaluation dates, unknown opponent version; not a confirmatory improvement test')
        return result
    except (OSError, ValueError, KeyError, TypeError):
        return dict(status='INVALID_OR_UNAVAILABLE_BASELINE', confirmatory=False)


def run_final_benchmark(experiment, output, *, max_seconds=86400, seed=39501,
                        historical_report=None, smoke=False, client_factory=None):
    if not math.isfinite(max_seconds) or max_seconds <= 0 or seed < 0:
        raise ValueError('Invalid benchmark limits')
    if smoke and client_factory is None:
        raise ValueError('Smoke benchmark requires an offline client; no live API calls')
    root, output = Path(experiment), Path(output)
    labels = report_labels(json.loads((root / 'continuation.json').read_text()))
    baseline_label, final_label = labels[0], labels[-1]
    policy, marker = verified_checkpoint(root, final_label, smoke=smoke)
    output.mkdir(parents=True, exist_ok=False)
    pilot_count, main_count = (4, 8) if smoke else (1000, 20000)
    report = dict(provider='slumbot', policy=final_label, sha256=marker['sha256'], smoke=smoke,
        status='STARTING', primary_metric='raw_bb', scores={'pilot': [], 'main': []},
        purpose='descriptive_final_policy_benchmark', alpha=.05,
        opponent_version_verified=False, estimate=None, historical_comparison=None)
    design = dict(policy=final_label, sha256=marker['sha256'], seed=seed, alpha=.05,
        pilot_hands=pilot_count, confirmation_hands=main_count, pilot_excluded=True,
        max_seconds=max_seconds, smoke=smoke, fixed_sample=True,
        interpretation='Cost-limited score estimate, not a guaranteed-powered test of improvement')
    write_json(output / 'design.json', design)
    # Fix the sample size before even the pilot; pilot estimates precision only.
    write_json(output / 'plan.json', design)
    journal = Journal(output / 'hands.jsonl')
    deadline = time.monotonic() + max_seconds
    try:
        import torch
        torch.set_num_threads(1)
        agent = Agent('checkpoint', checkpoint=policy, seed=seed)
        client = (client_factory or client_for)('slumbot', agent.action_menu)
        if client.provider != 'slumbot':
            raise ValueError('This benchmark accepts only Slumbot')
        journal.write('policy', label=final_label, sha256=marker['sha256'])
        report['opponent_version_before'] = version_for(client)
        for phase, count in [('pilot', pilot_count), ('main', main_count)]:
            # Independent action RNGs for pilot and confirmation.
            agent.rng = np.random.default_rng(seed + (phase == 'main'))
            report['status'] = phase.upper()
            write_json(output / 'report.json', report)
            for ordinal in range(count):
                if time.monotonic() >= deadline:
                    raise TimeoutError('Slumbot wall budget exhausted')
                score, _ = collect_hand(client, agent, journal, phase, final_label, ordinal)
                if not math.isfinite(score):
                    raise ProtocolError('Non-finite completed hand score')
                report['scores'][phase].append(score)
                if (ordinal + 1) % 100 == 0:
                    write_json(output / 'report.json', report)
            if phase == 'pilot':
                variance = float(np.var(report['scores']['pilot'], ddof=1))
                design['pilot_variance_bb2'] = variance
                design['estimated_95pct_half_width_bb100'] = 100*norm.ppf(.975)*math.sqrt(variance/main_count)
                write_json(output / 'plan.json', design)
        after, before = version_for(client), report['opponent_version_before']
        changed = before is not None and after is not None and before != after
        report.update(status='INVALID_OPPONENT_VERSION_CHANGE' if changed else 'COMPLETE',
                      opponent_version_after=after,
                      opponent_version_verified=before is not None and after == before)
    except TimeoutError:
        report['status'] = 'INCOMPLETE_TIME_LIMIT'
    except APIError as exc:
        # Transport error messages are sanitized; unresolved POSTs are never retried here.
        report.update(status='INCOMPLETE_PROVIDER_ERROR', error_type=type(exc).__name__, reason=str(exc))
    except (ProtocolError, ValueError, RuntimeError) as exc:
        report.update(status='EVALUATION_ERROR', error_type=type(exc).__name__)
    finally:
        main = report['scores']['main']
        if len(main) >= 2:
            report['estimate'] = estimate(main, complete=False)
            report['estimate']['conclusion'] = 'smoke_only' if smoke else (
                'descriptive' if report['status'] == 'COMPLETE' else 'incomplete_descriptive')
        if report['status'] == 'COMPLETE' and not smoke:
            report['historical_comparison'] = historical_comparison(historical_report, root, main)
        report.update(completed_pilot_hands=len(report['scores']['pilot']), completed_hands=len(main))
        journal.close()
        write_json(output / 'report.json', report)
        lines = [f'# {final_label} versus Slumbot', '', f"Status: {report['status']}",
                 f'Confirmation hands: {len(main)} / {main_count}. Pilot excluded.',
                 'Descriptive score only; no significance-based stopping.']
        if report['estimate']:
            result = report['estimate']
            lines += [f"Score: {result['bb_per_100']:.2f} bb/100; 95% interval {result['interval']}."]
        if report['historical_comparison']:
            lines += [f'Historical {baseline_label} comparison: ' + json.dumps(report['historical_comparison'])]
        (output / 'slumbot_report.md').write_text('\n\n'.join(lines) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--historical-report', type=Path)
    parser.add_argument('--max-seconds', type=float, default=86400)
    parser.add_argument('--seed', type=int, default=39501)
    if run_final_benchmark(**vars(parser.parse_args()))['status'] != 'COMPLETE':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
