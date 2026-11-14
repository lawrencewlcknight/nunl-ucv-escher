# Verification

## Production optimization v0.7.0 — 2026-11-15

* Final offline suite: **143 passed, 2 skipped** (49.31 seconds); skips are the
  opt-in Ray integration cases. Real two-worker tests: **2 passed** separately
  (50.01 seconds), including default/custom action menus and exact continuation.
* Replay-fill boundary tests cover minimum batches, floors, integer ceiling,
  full buffers and inference-only workers. Real cached/uncached fits verify
  Adam's actual update counters and preserve mature maximum budgets. Empty
  control fits do not advance target versions or temporal history.
* Critic bootstrap policies match collection for negative, tied, positive and
  tiny positive regrets, both fallback settings, legal masks and terminal rows.
* Large-label regression exercises actual gradient clipping before optimizer
  steps while preserving labels. Empty regret batches preserve learner state.
* A simulated interruption after a recovery-only checkpoint resumes with exact
  equality of networks, optimizers, replay and RNGs against uninterrupted
  training. Recovery-clock triggers and independent policy cadence are tested.
* Per-street row counters are checked in both sequential and real Ray collection.
  Tests use bounded smoke workloads; no configuration experiments, production
  training, cloud jobs or external opponent calls were run for this change.

Use `--policy-every` to control extraction. `--checkpoint-every` now controls
recovery-only saves; `--checkpoint-seconds` defaults to 1,800 seconds, checked
at completed iterations. Updated source/configuration requires a fresh run;
existing policy inference remains compatible.

## Encoder v5 validation — 2026-11-15

* Offline suite: **115 passed, 2 skipped**; skips are opt-in Ray tests.
* Real two-worker collection and exact continuation: **2 passed** separately,
  covering both the default and custom action menus.
* Tests distinguish strict current nuts from board ties and check the three new
  preflop equity interactions, exact call prices, and off-turn/postflop masking.
  Original named card and betting values remain unchanged.
* Frozen v4 metadata and nonzero saved-policy predictions are preserved. Current
  continuation passes; continuation from older layouts is rejected.
* Cross-repository audit: 30,000 ranked hands, 800 random card cases with 51 shared
  channels, all 169 preflop equity classes, and 612 decisions across 60 matched
  full hands agree. Shared fixtures pass independently in each repository.
* Single matched 100-hand/800-decision timing: v4/v5 policy-plus-critic encoding
  took 0.946/0.907 ms per decision on the first pass and 0.130/0.149 ms on the
  repeated pass. These local measurements include within-hand reuse and are
  not training-throughput or playing-strength evidence.
* Widths and fitting budgets are unchanged. Default online parameters total
  1,314,661; dense replay growth is 120 MB including calibration. Details and raw
  measurements: [ENCODER_V5_VALIDATION.json](docs/ENCODER_V5_VALIDATION.json).

## Encoder v4 validation — 2026-11-15

* Offline suite: **110 passed**, with two opt-in Ray tests skipped.
* Real Ray default/custom-menu collection and exact continuation: **2 passed**.
* Shared card fixture: 193 cases including every preflop suit class, identical
  to the fixture tested independently by the limit repository.
* Checks cover full next-card categories, off-menu aggression histories,
  hidden-card isolation, critic comparison, preflop table regeneration, global
  RNG isolation, v3 predictions and exact continuation after clearing caches.
* Version 0.5.0 wheel built without downloads, installed into a separate target,
  and completed a smoke iteration, policy fit and checkpoint reload from outside
  the checkout. The bundled preflop table and frozen v3 modules are included.
* Matched 100-hand/800-decision audit: policy-plus-critic encoding changed from
  0.913 to 1.000 ms on the first pass and 0.121 to 0.128 ms on the repeated pass.
  This measures feature construction only, with within-hand cache reuse; it is
  not a full training throughput or playing-strength result. Raw measurements,
  source fingerprints and replay costs: [ENCODER_V4_VALIDATION.json](docs/ENCODER_V4_VALIDATION.json).

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
Existing API equivalence and exact continuation tests use the v5 encoder.

Network tests check the default per-role shapes and parameter counts, independent
width/depth overrides, invalid configuration rejection, and policy reload with
both previous and current sizes. Continuation rejects changed architecture
settings. The Ray tests cover default sizes and nonuniform per-role overrides,
including bit-exact continuation. Smoke retains the full default network sizes.

For a small cost audit without training or network access:

```bash
python -m hunl_escher.feature_audit --hands 100
```

This compares the retained v2 exact channels, v3 and v5 encoders on the same 800 check/call
decisions, reports cold/repeated timings and replay allocation estimates, and
checks exact prefix retention. It is not a training-throughput benchmark.
