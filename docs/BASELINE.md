# Selected UCV-ESCHER baseline

## Decision

Use the **FHP Experiment 36 learner, continued unchanged in Experiment 38**:
the structured, cached, synchronous parallel UCV implementation with **2,500
TD updates for each of two critic members per outer iteration**. This is 5,000
critic optimiser steps in total, compared with 20,000 at the old 10,000-per-member
setting. It is the best-supported reduced-budget starting point, not a proven
optimum for no-limit Hold'em.

The extraction uses FHP source commit
`c2d7a5cf5c1bfeb98393fb185457b68c7854ce13`. The historical source has unrelated
working-tree edits in an evaluation launcher; none were needed or copied.
The inherited implementation traces through Experiment 9's frozen-target cache,
Experiment 7's parallel collector, and Experiment 2's structured networks and
compact float32 replay. Reusable checkpoint code was lifted out of experiment
packages into `hunl_escher`; runtime imports do not depend on historical experiments.

## Why 2,500 rather than 500

The completed **Leduc Experiment 51**, run `exp51-budget-20261101-220804`, compared
10,000, 2,500, 500 and zero critic updates. There were three paired seeds, 600
iterations and 12 million trajectories per run. The primary endpoint was mean
exact-average exploitability at iterations 450, 500, 550 and 600, in native
`NashConv / 2` units. Positive differences below mean worse than 10,000 updates.

| Updates per critic | Mean difference | 98.333% paired interval | Mean critic fit time | Mean learning time |
|---:|---:|---:|---:|---:|
| 10,000 | Reference | — | 17.447 h | 46.822 h |
| **2,500** | **+0.001656** | **[-0.000961, +0.004272]** | **4.906 h** | **37.218 h** |
| 500 | +0.000764 | [-0.012794, +0.014323] | 1.266 h | 36.951 h |
| 0 | +0.051951 | [+0.019309, +0.084592] | Approximately 0 | 29.948 h |

The 2,500-update arm met the study's exploratory acceptable-loss margin of 0.005;
500 did not because its uncertainty was much wider. Zero-Q was consistently
worse. The three-seed study does not establish equivalence or transfer to HUNL.
Runtime differences in other components prevent interpreting these totals as
isolated critic speedups; the 500-update arm barely improved mean total time
over 2,500 in this run. Its smaller critic cost alone is insufficient reason to
choose it as our baseline.

The compact result snapshot is in [critic_budget_evidence.json](critic_budget_evidence.json).
The original timing source is `analysis/costs_by_seed.csv` in the same run.
Experiment 50 studied temporal averaging; Experiment 52 studied advantage-label
noise. Neither is the critic-update-budget study. Four-fit averaging remains in
the inherited baseline for continuity; we do not claim that Experiment 50 proved
it beneficial.

## Learning specification

| Component | Selected setting |
|---|---|
| Algorithm | Outcome-sampling UCV-ESCHER with corrected all-action estimates and DCFR+ cumulative-regret fitting |
| Control coefficient | Fixed beta = 1 |
| Critics | Two cross-fit members; trajectory assignment determines held-out collection control |
| Critic fitting | Persistent online networks/Adam; 2,500 updates per member per iteration; batch 2,048; LR 0.001; gradient norm cap 10 |
| Target critics | Frozen during collection/fit; parameter average over the last four completed fits (available history during warmup) |
| TD target cache | Rebuilt once per eligible fit; no cache reused across target/replay/iteration changes |
| Sampling | Learned residual-standard-deviation proposal, mixed with 20% uniform legal-action probability mass |
| Calibration | Enabled; 2,000 updates per iteration; batch 2,048; LR 0.001; minimum variance 1e-5 |
| Regret fitting | One network per player; 750 updates per player per iteration; batch 2,048; LR 0.001; alpha 2.3; persistent cumulative networks |
| Instantaneous predictor | Disabled; prediction gate fixed at zero |
| Average policy | Grouped soft-target cross entropy; iteration weighting gamma 2; reset each fit; 20,000 updates; batch 2,048; LR 0.003 |
| Policy fit cadence | Checkpoints only, preserving the FHP approach; the initial runner uses completed-iteration checkpoints (50 by default), plus the final endpoint |
| Collection | 10,000 trajectories per player per iteration, 20,000 total; alternating player updates |
| Networks | Separate 64-wide card/context branches; 2×128 trunk for regret, critics and calibration; 2×192 trunk for average policy |
| Regret replay | Capacity 1,000,000 per player, reset for each player collection batch |
| Policy replay | 1,000,000-row reservoir |
| Critic replay | 1,000,000 transitions total, divided into two 500,000-row circular buffers |
| Calibration replay | 1,000,000 rows |
| Storage | Dense compact float32 replay, integer bookkeeping; no inherited FHP-only feature codebook |
| Parallel execution | Eight synchronous traversal actors as production starting point; deterministic budget partition; cached actor snapshots; central sequential learner fits |
| Development execution | One worker/one PyTorch thread by CLI default; real parallel integration test uses two actors |

**Leduc and FHP policy-fitting cadences differ.** Experiment 51 retained 20,000
policy updates every iteration, whereas the selected FHP system only fits its
policy at checkpoints. The new runner preserves checkpoint-only fitting and
restores training RNG state after each policy fit. Its iteration checkpoint
interval replaces the old experiment's six-active-hour schedule; it is configurable,
and should be selected using a measured HUNL iteration cost.

## HUNL-specific changes

The learning mechanism and budgets transfer; the FHP trained weights and its
flop-only feature contract do not. New policies start from fresh initialisation.

* Two players, 52 cards, two private cards each, board progression `0 3 1 1`,
  no-limit betting, 200bb stacks, 50/100 blinds, no rake.
* The initial FCPA interface has been replaced by OpenSpiel 1.6.3's `fullgame`
  engine and a configurable compact learner action menu. The default seven
  indices represent fold, check/call, min-raise, half-pot, pot, double-pot, all-in.
  Exact off-menu chip amounts can enter the state without translation to a menu
  size. OpenSpiel's bundled ACPC engine imposes a roughly 64-action-per-round
  safety ceiling; see [the interface specification](GAME_INTERFACE.md).
* Encoder v3 has **540 policy features and 641 critic features** with the
  default seven-action menu. Cards occupy separate hole/flop/turn/river channels
  with global suit canonicalisation. Compact public history preserves exact
  bet amounts and street boundaries. Explicit context includes observer, actor,
  street, stacks, total/street contributions, pot, call amount, raise bounds,
  legal-action mask and the exact size represented by each action index.
  The original 459/511-value prefixes remain intact; strategic features are appended.
* Critics additionally receive both hole-card sets. Policy/calibration/regret
  inputs never contain the other player's cards. Public turn/river order is
  preserved even where OpenSpiel's native public-card mask loses it.
* Four-street strategic descriptors now include exact current uniform-range
  strength, best-five category/kickers, structural draws, board texture, blockers,
  pot odds, effective stacks and action-specific costs. They are bounded,
  deterministic and cached. Current strength before the river is explicitly
  distinguished from showdown equity; no opponent range is inferred from it.
  See [STRATEGIC_FEATURES.md](STRATEGIC_FEATURES.md) for the review and cost audit.
* Continuation and policy contracts reject old FHP and initial FCPA checkpoints.
  Encoder v3 also rejects v2 policies and continuation states; new training is
  required. Full states are sharded with checksums and include actor/replay RNGs. Fits and
  checkpoints occur only at completed outer-iteration boundaries.

## Efficiency priorities

The cache avoids repeating frozen target inference thousands of times per fit.
Reduced critic updates, bounded actor scratch replay, compact learner arrays and
checkpoint-only policy fits are retained immediately. No historical paired
policy diagnostic, exact game-tree enumeration, old experiment orchestration or
automatic cloud launch is retained.

The next profiling pass must include collection, feature construction, critic
cache building, calibration, regret fitting, policy grouping/fitting, persistence,
peak resident memory and total wall time. Do not infer full-game training cost
from the tiny smoke run. Grouping may save less work in full Hold'em because
repeated information sets are rarer. Only then test 500 critic steps, more
selective policy extraction, richer bet menus, or accelerator changes against
quality per total compute. Neither training nor opponent API benchmarking has
been run at production scale in this repository.
