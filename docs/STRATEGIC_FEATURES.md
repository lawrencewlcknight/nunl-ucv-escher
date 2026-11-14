# Four-street strategic features, encoder v3

The default is `hunl_strategic_public_state_v3`. It adds poker structure to the
exact input rather than merging hands into strength buckets. Policy inputs grow
from **459 to 540** values; critics grow from **511 to 641**. The original prefixes,
suit symmetry, ordered board streets and complete public betting history remain
intact. New training is required: v2 policies and continuation states are rejected.

## Review and decision

The selected FHP Experiment 36/38 encoder used Experiment 33's AE package:
best-hand and board descriptors, exact uniform-opponent strength and betting
interactions. FHP ends on the flop, so its five-card strength is already terminal
strength. Hold'em needs best-five evaluation from five, six or seven cards and
separate representations of current strength and future possibilities. Fixed-limit
raise counts and FHP's preflop table cannot be transplanted into this game.

The historical feature comparisons do not justify adding every package. The
archived Experiment 29, 30, 31 and 33 reports all have direct-play confidence
intervals spanning zero against Experiment 10. Experiment 33's mean is -2.64
mbb/hand, with exploratory 95% interval [-15.71, 10.44], from three training seeds.
These compare entire trained packages, with different node counts and input
dimensions. They are not clean feature ablations or evidence of Hold'em optimality.
The reports are under the original FHP repository's
`cloud_outputs/exp{29,30,31,33}-features-*/evaluation/analysis/summary.json`.
The Leduc 51 evidence supports the retained critic budget, as documented in
[BASELINE.md](BASELINE.md); it does not establish the best Hold'em representation.

The implementation therefore retains a compact set with direct poker meaning.
It keeps the existing cross-fit UCV estimator, residual sampling, target caching,
network widths, 2,500 updates per critic and checkpoint-only average-policy fits.
No new learned auxiliary model, Monte Carlo feature estimator, range model or
expensive future-board equity search is introduced.

The distinction between current strength and hand potential follows the
[original Alberta hand-assessment work](https://poker.cs.ualberta.ca/publications/AAAI98.pdf).
Here future potential is represented by cheap structural draws, not a claimed
effective-hand-strength or action-conditioned equity calculation.

## Added information

Each own-card block has 49 named features; exact order is stored in checkpoint
metadata and `strategic_features.CARD_NAMES`.

| Family | Meaning |
|---|---|
| Starting hand | High/low rank, pocket pair, suitedness, rank gap, common straight window including the wheel, broadway fraction |
| Made hand | Nine category indicators and five ordered tie-break ranks for the current best five cards |
| Relative strength | Availability, exact win/tie/loss fractions, expected pot share and unbeatable indicator against every legal uniform opponent holding on the current board |
| Card removal | Fraction of board-compatible strongest holdings removed by own cards; remaining strongest-holding mass |
| Board and hand relations | Pairing, multiplicity, suit concentration, straight connectivity, high/low rank, river board category, playing the board, overpair/top-rank match/overcards, suit blockers |
| Draw structure | Exact next-card straight, flush and union completion fractions; flop runner-runner straight/flush fractions requiring both cards; next-card category improvement |

Uniform opponent enumeration contains 1,081 holdings on the flop, 1,035 on the
turn and 990 on the river after removing only own and board cards. The strongest
holding reference for blockers is computed before own-card removal. If own cards
block every holding in that reference class, remaining mass is zero; the feature
does not silently redefine that class to the next-best hand.

Before the river, current pot share is **not showdown equity**. Even on the river
it uses a uniform prior, not the opponent's betting-conditioned range. Preflop
strength is unavailable and explicitly zero with an availability flag; starting
hand descriptors remain available. No FHP preflop equity table is reused.

Draw probabilities count the 47 unseen flop cards or 46 unseen turn cards;
runner-runner features enumerate 1,081 unordered flop runouts. Already completed
straight/flush patterns have zero corresponding completion draws. A completion
can play the board or lose to an opponent's stronger hand/redraw. These are
structural possibilities, **not clean winning outs**. Draws are zero on the river.
Board pair/flush descriptors still inform the network of these qualifications.

The 32 betting features consist of 11 state descriptors and three per action in
the default seven-action menu:

* Actor/observer identity, postflop positional advantage, call price
  `call / (pot + call)`, fraction of stack needed to call, effective stack after
  calling relative to the resulting pot, logarithmic SPR, facing an all-in,
  and whether check/call closes this street.
* Current-strength interactions with call price, stack risk and effective stack.
  These are inputs to learn from, not hard-coded betting thresholds.
* For each legal action: incremental risk `cost / (pot + cost)`, the opponent's
  call price after that action, and the effective stack/pot share if called.
  Fold and unavailable actions have zero values. Exact native chip amounts are
  used, including minimum raises, off-menu histories and short all-ins.

Stack/pot uses the bounded monotone transform `stack / (pot + stack)`;
`log1p(SPR) / log(201)` supplies another scale useful in deep-stacked positions.
All descriptors fit [-1, 1]. No fixed-limit raise cap or FHP final-flop assumption
is used. The small blind's initial limp does not incorrectly mark the street as
closed: the big blind retains its option.

## Information boundary and architecture

Policy, regret and calibration features depend only on the observer's cards and
public state. Calibration sometimes observes a different player's decision;
call prices and menus still describe the actor, while own-strength interactions
are zeroed when observer and actor differ. This avoids pairing one player's hand
strength with another player's immediate decision as though they were the same.

Critics receive both players' card blocks, each calculated separately from that
player's own/public information, followed by the actor's betting features. The
actual opponent cards never narrow the uniform prior in either card block.
Terminal critic rows retain their zero placeholder and done mask.

Raw cards retain their existing 64-wide branch. The derived descriptors join
the public context branch, also 64-wide, before the existing trunk. Training,
Slumbot and GTO Wizard all invoke the same encoder. Dummy opponent cards used
for remote replay cannot enter the policy's card statistics.

## Cost and validation

There are bounded per-process LRU caches: 64 board tables and 8,192 card vectors.
Board tables share work across own holdings; suit-equivalent own/public card
sets share immutable feature vectors. Betting calculations remain inexpensive
and use current amounts. The evaluator is vectorised NumPy with no new dependency.
Features consume no random numbers and are stable across cold-cache restarts.

The recorded local ARM64/Python 3.12.2 microbenchmark covers 100 check/call hands,
800 decisions, both seats and every street. Combined policy-plus-critic encoding:

| Representation | Time per decision |
|---|---:|
| Original v2 prefix | 87 microseconds |
| v3 first pass through new hands | 941 microseconds |
| v3 repeated pass with card vectors cached | 123 microseconds |

The first pass already benefits from reuse within each hand; it is not an
all-cache-miss latency bound. The repeated pass fits in the card cache; production
hit rates may differ. This is feature construction only, not training throughput
or a claim that training is faster. The raw audit and commands are in
[FEATURE_VALIDATION.json](FEATURE_VALIDATION.json).

At inherited capacities, primary dense replay grows from 11.627 GB to 13.963 GB
(decimal), plus another 0.324 GB of calibration-feature storage compared with v2.
Primary replay excludes calibration, models, learner tensors, actor scratch,
caches and checkpoint copies. These are allocation estimates, not peak RSS.

Validation includes an independent scalar best-five oracle on 4,500 random
hands, 300 native OpenSpiel showdowns, rare-category tests, full legal-opponent
enumeration, exact draw combinatorics, all suit permutations, no hidden-card
leakage, bounded features and exact chip prices. The complete offline suite and
both real-Ray continuation tests pass. A fresh two-iteration smoke checkpoint
also passes 100 hands through both synthetic API protocols, with 1,142 identical
policy-probability checks, 64 off-menu raises, both seats and all four streets.
GTO Wizard remains offline-only until a key is available. Previous live Slumbot
validation belongs to the v2 interface snapshot in `API_VALIDATION.json`.

These checks support correctness and a bounded feature cost. They do not prove
an optimal strategy or improved playing strength. The implementation is the
current considered baseline for fresh training; any subsequent comparison should
measure quality against total compute, including this extra encoding and memory.
