# Verification

Use the pinned dependencies and Python 3.10–3.12.

```bash
python -m pytest -q
RUN_RAY_TESTS=1 python -m pytest -q -m ray
python -m hunl_escher.train --smoke --iterations 2 --output outputs/smoke
python -m hunl_escher.train --smoke --iterations 3 --output outputs/smoke --resume
```

The suite covers the four-street no-limit game contract, pot/all-in/fold rules,
zero-sum settlements, suit symmetry, hidden-card isolation, retention of public
turn/river order, bounded monetary features, masked seven-action unbiased estimation,
sampling support, checkpoint policy probabilities, cached/uncached learner
agreement and exact continuation through replay, optimisers and critic history.
The opt-in Ray tests check real worker collection and exact parallel continuation
with both the default seven-action menu and the configurable four-action menu.
The interface tests replay 80 seeded hands against a separate native OpenSpiel
state, including arbitrary integer-chip bets, and verify legal actions, chip
conservation and identical payoffs. They also cover initial blind accounting,
rounding/deduplication, short all-ins, exact street-to-hand conversion, nonacting
traverser observations, a long multi-street minimum-raise history, clone isolation
and rejection of mismatched action-menu checkpoints.

Smoke fits actual networks and writes actual checkpoints using reduced budgets.
It does not establish strategy quality, full-budget throughput, cloud deployment
readiness or integration with either external benchmark API.

## API interface

`tests/test_remote.py` adds synthetic protocol equivalence, hidden-card-safe
reconstruction, exact policy inference, mocked HTTP sessions and timeout recovery.
Run a standalone offline verification with:

```bash
python -m hunl_escher.play --opponent offline --hands 100 --output outputs/offline.jsonl
```

The normal pytest suite makes no network requests. Live tests are explicit CLI
commands only. See [API_INTERFACE.md](docs/API_INTERFACE.md) for the bounded
Slumbot smoke and the GTO Wizard key requirement.

## Strategic features

`tests/test_strategic_features.py` checks 4,500 random 5/6/7-card hands against
an independent best-five oracle, 300 OpenSpiel showdowns, rare hand categories,
full opponent enumeration, blockers, wheel/open-ended/backdoor draws, suit
permutations, exact no-limit action prices, off-turn observations and v2 rejection.
Existing API equivalence and exact continuation tests use the v3 encoder.

For a small cost audit without training or network access:

```bash
python -m hunl_escher.feature_audit --hands 100
```

This compares the retained v2 prefix with v3 features on the same 800 check/call
decisions, reports cold/repeated timings and replay allocation estimates, and
checks exact prefix retention. It is not a training-throughput benchmark.
