"""Long continuations retain complete checkpoint families and error control."""
import json
from pathlib import Path

import pytest

from hunl_escher.continuation import report_labels
from hunl_escher.continuation_schedule import continuation_labels


def test_historical_reports_stay_readable_and_long_reports_are_explicit():
    assert report_labels({'baseline_hours': 200}) == ('P200', 'P206', 'P212', 'P218', 'P224')
    labels = continuation_labels(224, 72)
    assert len(labels) == 13 and labels[0] == 'P224' and labels[-1] == 'P296'
    assert report_labels(dict(baseline_hours=224, additional_hours=72, labels=labels)) == labels
    with pytest.raises(ValueError, match='labels'):
        report_labels(dict(baseline_hours=224, labels=labels))


@pytest.mark.parametrize('duration', [0, -24, 6, 48, 73, 72.0, True])
def test_unsupported_duration_is_rejected(duration):
    with pytest.raises(ValueError):
        continuation_labels(224, duration)


@pytest.mark.parametrize('primary_improved', [False, True])
def test_long_production_evaluation_controls_all_twelve_secondaries(tmp_path, monkeypatch, primary_improved):
    from hunl_escher import continuation_evaluate as evaluator
    root = tmp_path / 'training'; root.mkdir()
    labels = continuation_labels(224, 72)
    (root / 'continuation.json').write_text(json.dumps(dict(
        baseline_hours=224, additional_hours=72, labels=labels)))
    monkeypatch.setattr(evaluator, 'verified_checkpoint', lambda *args, **kwargs: None)
    calls = []
    def child(command, log, **kwargs):
        def arg(flag): return command[command.index(flag)+1]
        alpha = float(arg('--alpha'))
        calls.append((arg('--a'), arg('--b'), alpha, kwargs['timeout']))
        path = Path(arg('--output'))
        path.mkdir()
        result = dict(status='COMPLETE', result=dict(confidence=1-alpha,
            bb_per_100=15., interval=[5., 25.], confirmatory=True,
            conclusion='improved' if len(calls) > 1 or primary_improved else 'inconclusive'))
        (path/'report.json').write_text(json.dumps(result))
        return dict(returncode=0, timed_out=False)
    monkeypatch.setattr(evaluator, 'run_child', child)
    report = evaluator.run_suite(root, tmp_path/'evaluation', max_seconds=43200, external_benchmarks=False)
    assert len(calls) == 13 and calls[0][2] == .05
    assert sum(c[2] for c in calls[1:]) == pytest.approx(.04)
    assert [c[2] for c in calls[1:]] == pytest.approx([.04/12]*12)
    assert report['primary'] == 'P296-vs-P224'
    assert report['secondary_family_size'] == 12
    assert report['primary_passed'] is primary_improved
    for item in list(report['head_to_head'].values())[1:]:
        assert item['result']['confirmatory'] is primary_improved
    text = (tmp_path/'evaluation/continuation_report.md').read_text()
    assert '12 adjacent comparisons: 99.667% intervals' in text
    assert 'P296-vs-P290' in text and 'Four adjacent' not in text
