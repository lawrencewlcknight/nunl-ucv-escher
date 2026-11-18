# HUNL strategic-position evaluation

This offline suite lives in `hunl_escher/strategic_audit`. It evaluates trusted
playable policy checkpoints without training, API credentials or a cloud job.
It follows the FHP suite's distinction between provable mistakes and
opponent-dependent opportunities, adapted to 200bb heads-up no-limit Hold'em.

## What is measured

**Exact river decision costs.** For each public position and hero holding, all
990 compatible opponent holdings are integrated exactly. Every legal learner
root action is evaluated, including its exact chip amount. The candidate's own
subsequent decisions retain its deployed action probabilities. Hidden hands are
integrated **before** choosing the best root action: no player receives hidden
cards. `local_gap_bb` is the best root action EV minus the deployed EV. These are
local deviations, not a recursively optimized best response or exploitability.

Three frozen synthetic stress panels are evaluated separately:

| Panel | Stipulated river range | Subsequent opponent behaviour |
|---|---|---|
| calling_station | Uniform compatible hands | Always check/call |
| value_pressure | Weight `exp(4*(strength-.5))` | Mostly jams strong hands, seldom bluffs |
| polarized_pressure | Weight `.1 + 8*(strength-.5)^2` | More weak-hand jams, strong value jams |

Here strength is the opponent's exact showdown equity against uniform hands
compatible with **its own** cards and the public board. It does not use the hero's
actual cards. Before hero blockers, this quantity is calculated for every
board-compatible holding. At the root, remove hero-blocked holdings and normalize
the stipulated weights. These are explicit conditional priors, **not** ranges
inferred from the preceding betting history, and not GTO or Slumbot ranges.

The pressure policies use only fold/check/call/all-in. Their jam probabilities
are .90 for strength >= .85 and .05 otherwise (value), and .85 for strength >= .80,
.35 for strength < .25, .02 otherwise (polarized). Facing a bet, conditional on
not jamming, they call with probability .90 if their uniform equity meets the
public pot-odds threshold, .10 otherwise. With no bet they check if not jamming.
The all-in probability is zero when no raise is legal. These intentionally simple
stress opponents make **all** remaining branches inexpensive to integrate. The
candidate retains its entire checkpoint action menu; no candidate actions are
removed or branch depths truncated. This panel is narrower than the FHP frozen
learned-opponent panel; it is not a substitute for real opponent benchmarks.

**Provable dominance.** When a river call closes the hand, folding is flagged
only if calling earns strictly more against *every* compatible opponent holding.
This includes strict nuts and board-only ties where folding forfeits sunk chips.
Calling is flagged universally only if folding strictly beats it for every
holding. `guaranteed_cost_bb` uses the minimum loss over those holdings, multiplied
by the probability of the dominated action. It is a lower bound, not another
cost to add to the local gap. Range-conditioned call advantage and exact call
threshold are reported separately. Checking the nuts and bluffing air are never
blanket failures.

**Exact all-in draw decisions.** Two additional turn probes (nut draw and made
flush facing a shove) enumerate all 46 possible rivers and all 990 compatible
opponent holdings for each, or 45,540 joint outcomes per position. Exact uniform
range equity, pot-odds thresholds, call advantage and deployed decision cost are
reported separately. The engine also supports imported flop fold/call all-ins
(1,070,190 joint outcomes, substantially slower); preflop all-ins are not solved.
These calculations assume uniform unseen cards, not an inferred shove range.

**Earlier-street behaviour.** Twelve preflop/flop/turn probes expose premium and
weak openings, 3-bet responses, overpairs, bluff catchers, nut draws, combination
draws and made draws. They report action probabilities, aggression and entropy.
They do **not** assign preferred actions or estimate future-street EVs. This
avoids presenting current made-hand strength as eventual showdown equity.

**Implementation checks.** Every position reconstructs the same public view with
different hidden placeholder cards. Every root checks private-card order and
flop-card order. The first position in each family is queried under all 24 suit
permutations. Probability violations or hidden-card view changes fail the audit;
order/suit total-variation differences are reported, not silently repaired.
A raw-input network may have learned imperfect symmetry; the canonical encoder
should have negligible drift. Turn and river ordering is preserved.

## Coverage and comparisons

The default seed freezes **526 positions: 374 development and 152 assessment**.
There are 13 river board families in total: eight base development boards, one
additional kicker-contrast board, and four assessment boards. Representative
holdings span each available made-hand category plus a strongest holding.
Eight histories cover both seats, limped and 3-bet pots, checking, half-pot,
pot-sized and double-pot bets, river all-ins and multiple barrels. Explicit
blocker and kicker contrasts share histories and boards. Suit names are randomly
assigned once when the base library is generated, independently of candidates.

All checkpoints see identical cases and stress panels. Scores first average
positions within a family, then give families equal weight; this is a balanced
stress suite, **not natural game visitation**. The report includes costs in bb
per audited decision, cost/pot ratios, and overlapping category summaries.
Never interpret these numbers as bb/100 hands or sum costs along a trajectory.

Matched comparisons show B minus A in local gap (negative is favourable) and
conditional deployed EV (positive is favourable). Each candidate keeps its own
later continuation, so these compare the deployed policies' local decision
quality, not just their root logits. Three panels remain separate; there is no
arbitrary composite GTO score. Tags overlap and are not independent samples.
Exact conditional enumeration has no Monte Carlo error, so the suite fabricates
no confidence interval from repeated positions or suits. Generalization beyond
these chosen boards/opponents, and training-seed variability, remain unmeasured.

Assessment is locked behind an explicit flag and separate output namespace.
Finish model selection on development first. Repeatedly tuning to assessment
consumes that holdout. The split guards audit-driven selection; it does not
claim that self-play training never encountered these cards.

## Smoke and full development runs

From the HUNL repo in its Python environment:

```bash
python -m hunl_escher.strategic_audit library --smoke \
  --output outputs/strategic-smoke-library.json
python -m hunl_escher.strategic_audit run \
  --library outputs/strategic-smoke-library.json \
  --policy call=rule:check_call --policy random=rule:random \
  --output outputs/strategic-smoke --max-seconds 60 --threads 1

python -m hunl_escher.strategic_audit library \
  --output outputs/strategic-library-v1.json
python -m hunl_escher.strategic_audit validate \
  --library outputs/strategic-library-v1.json
python -m hunl_escher.strategic_audit run \
  --library outputs/strategic-library-v1.json \
  --policy P8=/absolute/path/to/P8.pkl \
  --policy P32=/absolute/path/to/P32.pkl \
  --policy P56=/absolute/path/to/P56.pkl \
  --output outputs/strategic-development --max-seconds 600 --threads 1
```

Only the roughly megabyte-sized playable policy files are needed, not recovery
states, replay arrays or optimizers. Model metadata selects the saved encoder and
architecture, including supported historical encoders. Compared policies must
share their action menu. Rule controls use the default seven-action menu.

After model selection, repeat the run with a **new output directory** and
`--partition assessment --unlock-assessment`. Earlier-street and turn all-in probes are currently
in development only; assessment measures the separate river board families.
There are no API calls or automatic cloud submissions anywhere in this suite.

## Runtime, integrity and output

One CPU thread is the default, with one checkpoint loaded at a time. Cached board
rank/equity tables and vectorized hidden-hand integration avoid querying the
network once per possible opponent holding. Use a smoke run to measure the
actual machine before assigning a larger budget.

`--max-seconds` is a cooperative invocation budget, checked between positions,
suit queries and tree nodes; a single inference/load/board-table calculation can
overrun it. It is not an OS-enforced VM hard stop. `INCOMPLETE_TIME_LIMIT` retains
completed position records and suppresses checkpoint comparisons. There are no
automatic retries. Resume explicitly with the same command and `--resume`; a
larger time budget is permitted. Checksummed records are reused. Changed library,
policies, source, runtime versions, partition or thread count require a new output
directory. A stale `audit.lock` after forced process death must only be removed
once the prior process is confirmed stopped.

Outputs:

- `manifest.json`: frozen library, checkpoint hashes/metadata, game contract,
  source fingerprint, runtime, weighting, partition and stress-panel contract.
- `records/*.json`: atomic checksummed results for every completed position and
  policy, including exact root values, action costs, probabilities and checks.
- `report.json`: completion status, coverage, per-policy/per-panel/category
  summaries and matched checkpoint changes.
- `report.html`: self-contained browser report with largest-gap examples and all
  position records. Extreme examples are not representative average behaviour.

The suite complements duplicate-deal checkpoint matches, Slumbot and GTO Wizard.
Its purpose is to reveal *where* a policy is making costly decisions, not to
certify equilibrium play or replace statistically powered whole-game evaluation.

## Initial validation — 18 November 2026

The focused audit, hand evaluator, game interface and remote interface regression
suite passed **95 tests**. Audit tests cover independent native OpenSpiel branch
settlements in both seats, all 45,540 native opponent/river settlements for a turn
all-in draw, known nut/tie cases, integration before maximization, hidden-card
independence, illegal probability rejection, real checkpoint inference, suit/card
order, locked assessment, checksum corruption, time limits, explicit resume and
interruption status.

A development-only run of the stored P32 policy (SHA-256
`73eec96257ba92f0c6b8e4fb43f040d8f88a7eb80c589dcffd71f567bb49f129`)
completed all **374 development positions in 7.08 seconds** on this laptop with
one Torch thread. This is one measured run, not a guaranteed runtime on other
machines. All measured card-order/suit probability differences were zero.
Assessment was not evaluated for P32.

The report is in `outputs/strategic-audit-v1-20261118/p32-development/` and the
frozen library is its sibling `library.json`. In one board-only royal-flush case
P32 folds with probability 0.9538746 against a river shove, even though calling
guarantees a split. With 3bb already invested, that loses 2.861624bb in expected
value at that decision. This is a concrete diagnostic error, not an estimate of
its frequency or impact in normal games. No training parameters were changed.
