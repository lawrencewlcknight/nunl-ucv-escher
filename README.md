# HUNL UCV-ESCHER

A clean research repository for training Unbiased Control-Variate ESCHER in
heads-up no-limit Texas Hold'em. The learner is extracted from the efficient
FHP Experiment 36/38 implementation, with **up to 2,500 updates per critic per fit**,
scaled to occupied replay during warm-up.
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

Training reuses encoded features within each unchanged game state, keyed by
encoder and observing player. Every action clears the cache; returned arrays
remain independently mutable. Binary suit ordering uses packed bits, inference
uses explicit tensor devices, and frozen-target construction takes contiguous
replay views while preserving final-batch padding and skipping unused fields.
Both Hold'em variants also replace repeated rank sorting with an exact, read-only
8,192-mask lookup (832 KiB per process); tests exhaust all masks and compare
30,000 random hands against the previous sorting operation.
These execution optimisations preserve the feature contract and learning
configuration. `tests/test_training_efficiency.py` compares replay, RNG, model
and Adam states with the previous operations, including production minibatches.
Existing source-fingerprint checks for training continuation still apply. Worker
pipelining and persistent replay-hash caches remain deferred until VM profiling
supports them.

This is a runnable development learner, not a trained competitive model.
Slumbot and GTO Wizard API adapters, inference-only play and offline equivalence
tests are implemented in [the API interface](docs/API_INTERFACE.md). Slumbot has
a small live smoke test; GTO Wizard remains verified offline pending an API key.
The [four-plus-four-hour learning check](docs/LEARNING_EXPERIMENT.md) implements
checkpoint restart, paired head-to-head statistics and isolated external benchmarks.
Production playing strength has not yet been measured.

## Install

For a controlled sampler comparison using saved replay, see the
[policy sampler experiment](docs/POLICY_SAMPLER_EXPERIMENT.md). It refits matched
policies without new self-play, evaluates them locally, and provides a separate
Slumbot follow-up after a positive primary result. Production defaults are unchanged.

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

## Continue training from a checkpoint

Full recovery checkpoints support continued training. Re-run the original
training command with the same output directory and options, add `--resume`, and
set `--iterations` to the desired **total** iteration count. For example, after
training the eight-worker baseline to iteration 100:

```bash
python -m hunl_escher.train --iterations 200 --workers 8 --threads 8 \
  --policy-seconds 21600 --recovery-seconds 21600 \
  --seed 0 --output outputs/baseline-seed0 --resume
```

The runner automatically loads the recovery directory referenced by `latest.json`
and continues at the next iteration. It restores all online/target networks,
Adam states, persistent replay, critic target history, counters, random-number
states (including workers), and recovery/policy scheduling. Fresh regret batches
and temporary caches are rebuilt as they are during uninterrupted training.

Keep the whole run directory: `manifest.json`, `latest.json`, and the latest
`state_*` directory with all its shards. Shards are verified before loading.
Use the same source, configuration, Python/dependency versions, seed, worker
count, thread count and scheduling options. The total iteration limit and the
additional active-time stopping budget may change. If the iteration target has
already been reached, resume performs no further work.

`policy_*.pkl` files are inference exports; they do not contain the replay and
optimizer state required for full training continuation. Every policy export
also produces a full recovery snapshot, subject to latest-only recovery retention.

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
  --policy-seconds 21600 --recovery-seconds 21600 \
  --seed 0 --output outputs/baseline-seed0
```

This is a substantial workload: 10,000 trajectories **per player per iteration**,
large replay buffers, and 20,000 policy updates per policy extraction. Check memory
and throughput on the intended training machine before choosing a long run.
The command uses the current machine and does not submit any cloud jobs.

Automatic recovery checks run every **21,600 wall-clock seconds (six hours)**,
on a fixed grid from run start. A check skips saving if a successful recovery
checkpoint was completed less than **3,600 seconds (one hour)** ago. This includes
recovery snapshots saved alongside policy fits. At exactly one hour, saving is
eligible again. A skipped check advances to the next six-hour slot; it does not
retry when the one-hour window expires. For example, a save at 5.5 hours suppresses
the six-hour check, with the next check at 12 hours.

Checks run after completed iterations, so long iterations can delay them.
Downtime counts as wall time; missed slots cause at most one check on resumption.
The grid, last successful save time and skipped checks survive restart. A skipped
check updates only the small latest-pointer metadata, without rewriting replay.
Policy fitting remains every **21,600 active learner seconds**, excluding policy
fitting, persistence and downtime. Policy-triggered, explicit iteration and final
saves remain unconditional.

`--checkpoint-every N` adds recovery-only saves every N iterations (default 0).
`--policy-every N` adds policy fits every N iterations (production default 0;
smoke default 1). Set `--policy-every 0` for time-only policy scheduling in smoke.
`--recovery-seconds 0` disables timed recovery; `--checkpoint-seconds` is an alias
for `--recovery-seconds`. Policy fits also save full recovery state, including the
selected policy and matching optimizer. The final requested iteration always
fits a policy and saves recovery state. Both schedule clocks survive resume;
recovery-only saves never fit a policy. No background timer interrupts learning.

`--resume` checks
the source fingerprint, runtime, seed, configuration and worker arrangement;
`--iterations` is the desired **total**, not additional iterations. Use trusted
checkpoint files only; their PyTorch/pickle metadata is not an untrusted format.
Only the newest recovery checkpoint is retained. After a new save passes checksum
verification and `latest.json` is updated, all older recovery checkpoints in that
run directory are deleted. Failed saves preserve the previous recovery checkpoint.
Playable `policy_*.pkl` files and metrics are retained separately. Allow room for
the old and new recovery checkpoints during saving. Cloud orchestration remains
future work.

Both Hold'em implementations use 750 regret updates/player, replay-scaled
250–2,500 critic updates and 100–2,000 calibration updates, regret target caching,
and gradient norm clipping at 10. Historical policy reservoirs hold eight million
rows for no-limit and four million for fixed-limit Hold'em.
Policy fitting uses a 5% grouped holdout to select within one 20,000-update fit.
See the [shared training specification](docs/TRAINING_ALIGNMENT.md) for the exact
settings and deliberate game-size differences. These changes require a fresh
training run; previously saved policies remain usable for inference.

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

## Continue P8 for another 24 hours

See the [continuation experiment protocol](docs/CONTINUATION_EXPERIMENT.md) for
full-state recovery, six-hour P14/P20/P26/P32 policies, final reload verification,
five local head-to-head comparisons and a final-policy-only Slumbot benchmark.
The training and Slumbot launchers are separate: the small evaluator can be
started only after verified P32 publication, avoiding an idle paid VM.
Launchers require `--execute` to create cloud resources.

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
