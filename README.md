# HUNL UCV-ESCHER

A clean research repository for training Unbiased Control-Variate ESCHER in
heads-up no-limit Texas Hold'em. The learner is extracted from the efficient
FHP Experiment 36/38 implementation, with **2,500 updates per critic per fit**.
See [the baseline specification](docs/BASELINE.md) for the choice and evidence.

The game uses OpenSpiel 1.6.3's full no-limit engine: two players, a full 52-card
deck, preflop/flop/turn/river, 50/100 blinds, and 20,000-chip stacks reset every hand.
A versioned Python adapter supplies exact chip amounts, public history and
private-card-safe neural inputs. The default learner menu is fold, check/call,
minimum raise, half-pot, pot, double-pot and all-in; illegal and duplicate sizes
are masked. Exact off-menu opponent bets are accepted by the state interface.
See [the game interface specification](docs/GAME_INTERFACE.md) for action IDs,
chip conventions, features, engine limitations and an executable example.

The [v5 strategic encoder](docs/STRATEGIC_FEATURES.md) adds current hand strength,
draws, board texture, blockers and no-limit betting economics while retaining
the exact original information. V4 adds next-card categories, public aggression,
cached preflop equity and critic-only hand comparisons. V5 adds strict current nuts
and explicit preflop equity/call-cost interactions. It uses bounded caches;
training starts fresh, while saved v3/v4 policies remain loadable for inference.

The [network configuration](docs/NETWORK_CONFIGURATION.md) uses 3×256 trunks for
regret and average policy, 2×192 for critics and 2×128 for residual calibration,
with separately sized exact-card, strategic-card and context branches. The six online models total
1.31 million parameters. These defaults require a fresh training run; saved v3/v4
policies with the previous widths remain usable for inference.

This is a runnable development learner, not a trained competitive model.
Slumbot and GTO Wizard API adapters, inference-only play and offline equivalence
tests are implemented in [the API interface](docs/API_INTERFACE.md). Slumbot has
a small live smoke test; GTO Wizard remains verified offline pending an API key.
Large-scale strength evaluation remains future work.

## Install

Use Python 3.10–3.12 (3.12 is the development reference):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,parallel]'
```

Core dependencies are pinned in `pyproject.toml`. The parallel extra adds Ray;
sequential training does not need it. The source repository and copied-file
hashes are recorded in [provenance.json](docs/provenance.json). Upstream attribution
and the original licence-status note remain in [UPSTREAM.md](vr_deep_cfr/UPSTREAM.md).

## Verify and run

```bash
python -m pytest -q
python -m hunl_escher.train --smoke --iterations 2 --output outputs/smoke
python -m hunl_escher.train --smoke --iterations 3 --output outputs/smoke --resume
RUN_RAY_TESTS=1 python -m pytest -q -m ray
```

Smoke reduces replay, minibatch and optimiser budgets; it is not evidence of
playing strength. Tests exercise real OpenSpiel and PyTorch. Ray tests start
real local worker processes and are opt-in.

The inherited learning budget is in
[`hunl_escher/data/baseline.json`](hunl_escher/data/baseline.json).
An explicit training invocation using the original eight-worker arrangement is:

```bash
python -m hunl_escher.train --iterations 100 --workers 8 --threads 8 \
  --checkpoint-every 50 --seed 0 --output outputs/baseline-seed0
```

This is a substantial workload: 10,000 trajectories **per player per iteration**,
large replay buffers, and 20,000 policy updates per checkpoint. Benchmark memory
and throughput on the intended training machine before choosing a long run.
The command uses the current machine and does not submit any cloud jobs.

Each checkpoint saves a playable policy and a checksummed, sharded continuation
state containing networks, optimisers, replay, temporal critic history and RNGs,
including actor RNGs. Only complete outer iterations are saved. `--resume` checks
the source fingerprint, runtime, seed, configuration and worker arrangement;
`--iterations` is the desired **total**, not additional iterations. Use trusted
checkpoint files only; their PyTorch/pickle metadata is not an untrusted format.
All checkpoint directories are retained; provision disk space for the chosen
cadence. A retention policy and cloud runner remain future work.

## Play and verify the API adapters

```bash
python -m hunl_escher.play --opponent offline --hands 100 --output outputs/offline.jsonl
python -m hunl_escher.play --opponent slumbot --hands 10 --output outputs/slumbot.jsonl
```

The first command makes no network requests. The second plays anonymously with
check/call. Add `--agent checkpoint --checkpoint PATH` to use a saved policy.
For GTO Wizard, set `GTO_WIZARD_API_KEY` securely and use `--opponent gtow`.
Install the `api` extra for the CA-bundle fallback. See the [API guide](docs/API_INTERFACE.md)
for exact units, resumption limits, logs and credential handling.

## Layout

| Directory | Purpose |
|---|---|
| `hunl_escher/` | Game, four-street encoder, configuration, CLI and checkpoints |
| `unbiased_escher/` | UCV estimator, structured learners, compact replay, parallel collection |
| `adaptive_escher/` | Persistent frozen-target critic and target cache |
| `vr_deep_cfr/` | Inherited neural cumulative-regret and replay foundations |
| `tests/` | HUNL rules, information boundaries, cache equivalence, resume and actors |
| `docs/` | Baseline evidence, provenance and next implementation steps |

No old FHP experiment runners, results, checkpoints, cloud launchers, evaluation
suite or Git history were copied. The original FHP repository is untouched.
