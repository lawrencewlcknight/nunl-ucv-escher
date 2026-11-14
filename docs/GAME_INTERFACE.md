# OpenSpiel HUNL interface v2

OpenSpiel remains the rules engine. The learner receives a Python adapter around
`universal_poker` with `bettingAbstraction=fullgame`. It delegates dealing,
legal betting, state transitions and showdown settlement to the pinned
OpenSpiel 1.6.3 engine. The UCV estimator, two cross-fit critics, 2,500 critic
updates per member, four-fit frozen targets and all optimisation budgets remain
unchanged. Interface and model dimensions change; initialise new models.

## Rules and identities

Two players, 52 cards, two private cards each, flop/turn/river of 3/1/1 cards,
50/100 blinds, 20,000 chips per player, no ante or rake, stacks reset each hand.
OpenSpiel player 0 is the button/small blind and acts first preflop; player 1 is
the big blind and acts first postflop. Slumbot's position numbers have the
opposite meaning; a future API adapter must translate these explicitly.

`HUNLGame`/`HUNLState` implement the methods used by the learner. They are Python
adapters, not registered `pyspiel.Game`/`pyspiel.State` subclasses. Do not pass
them directly to C++ evaluation routines expecting native states. Native tensors
and `native_history()` are diagnostic APIs; the learner uses the feature encoder.

## Three distinct action representations

| Method | Meaning |
|---|---|
| `state.apply_action(index)` | At a decision, apply a learner menu index. At chance, apply an OpenSpiel card ID. |
| `state.native_action(index)` | Translate a legal learner index to the exact native action without mutating. |
| `state.apply_native_action(action)` | 0 fold, 1 check/call, N>=2 raise **total hand contribution to N chips**. At chance, card ID. |
| `state.apply_street_raise_to(amount)` | Raise **total contribution on the current street to amount**. |

Illegal inputs are rejected before calling the native mutation. Native IDs and
menu IDs must never be interchanged. `child(index)` clones and applies a menu
action. There is deliberately no ambiguous `full_history()` adapter method:
`native_history()` returns native IDs, which must be replayed with
`apply_native_action()`.

The default network output indices are:

| Index | Meaning |
|---|---|
| 0 | Fold |
| 1 | Check/call |
| 2 | Minimum legal raise |
| 3 | Half-pot raise |
| 4 | Pot raise |
| 5 | Double-pot raise |
| 6 | All-in |

For pot fraction f, the native raise-to amount is
`max(total_contributions) + round_half_up(f * (pot + amount_to_call))`.
Sizes outside OpenSpiel's legal range are masked, not clipped. Duplicate amounts
use the first menu index, except the all-in index reserves the maximum raise.
For example, initially min-raise and half-pot both mean 200 chips; only min-raise
is enabled. The opening pot raise is 300, not 250. An all-in call is check/call.

The JSON config's `action_menu` can select/reorder the supported names; it must
start with fold/check_call and include all_in. For an initial-baseline-sized
network use `["fold", "check_call", "pot", "all_in"]`. Driver, actors, encoder,
policy checkpoints and continuation contracts all carry the same ordered menu.
Changing it requires fresh training; incompatible checkpoints are rejected.

A small menu is an abstraction of our available actions. The engine still
accepts exact off-menu actions and the next neural input retains their true
amounts. This does not by itself establish good generalisation to such histories.

## Explicit information boundary

`state.decision_view(player=None)` returns an immutable `DecisionView`: observer,
acting player, observer's two cards, ordered public board, street, both stacks,
total and street contributions, actual pot, acting player's call amount, legal
raise bounds, action mask, each menu action's native amount, and public history.
No opponent private cards enter this view. With no argument, observer=actor.
UCV residual calibration also requests the traverser's view at opponent nodes;
the observer can therefore differ from the actor. Legal actions and call amount
always describe the actor's current decision.

The encoder builds suit-canonical hole/flop/turn/river channels using only that
observer's visible cards. Training critics use a separate method to obtain both
hole-card sets and canonicalise those together with the board. Terminal critic
rows contain a zero placeholder, payoff and done mask, and are not bootstrapped.

Encoder v3 dimensions are **540 policy/regret/calibration information features**
and **641 critic full-state features**. Calibration also appends its inherited
algorithm-specific inputs. Within the policy vector, card channels occupy 208
values; within the critic vector they occupy 260. The original common 251-value context
is ordered as follows:

1. Observer one-hot (2), actor one-hot (2), street one-hot (4).
2. Both stacks (2), total contributions (2), street contributions (2), pot (1),
   call amount (1), minimum/maximum native raise-to (2).
3. Public event count (1), legal menu mask (7), native raise sizes per index (7).
4. Full chronological public event sequence, padded to 218 events.

Chip inputs are divided by 20,000, except pot by 40,000. Event count is divided
by 218. History uses one float per event: chance=0, fold=-1, check/call=-0.5,
raise=exact total contribution/20,000. Chance card identities are removed; their
positions preserve street boundaries. Event count distinguishes real events
from padding. Together with fixed dealing order and heads-up turn order this
retains the public action sequence. Float32 preserves distinctions between all
integer chip amounts in this game. Overflow is rejected, never truncated.

The original 459/511 prefixes are followed by 49 own-card strategic descriptors
(policy) or 49 per player (critic), then 32 betting descriptors. These include
current hand strength, draw structure, board texture, card removal and no-limit
action economics. See [STRATEGIC_FEATURES.md](STRATEGIC_FEATURES.md) for semantics,
information boundaries and the distinction between strength and runout equity.

A native three-value-per-event history would need 654 values. The compact
218-value history keeps replay smaller without dropping earlier streets. Full
primary replay grows from 11.63 GB with the v2 features to **13.96 GB** with v3 at the
inherited million-row budgets (three policy/regret buffers plus the critic
ensemble total). This excludes calibration, caches, sampled tensors, models,
actor scratch and checkpoints. These are allocated-array estimates, not measured
peak RSS. Profile mature replay before launching a full-budget job.

Worker traversal chunks are capped so worst-case scratch records stay near
28,800 per chunk (131 traversals at the fullgame length bound), rather than
multiplying the previous scratch capacity by the longer hand bound. Chunking
does not change the requested number of trajectories per iteration.

## Example

```python
from hunl_escher.game import load_hunl_game
from hunl_escher.features import HUNLFeatureEncoder

game = load_hunl_game()
state = game.new_initial_state()
for card in (0, 5, 10, 15):  # deterministic engineering example only
    state.apply_action(card)
assert state.decision_view().pot == 150
state.apply_street_raise_to(275)  # exact opening raise, outside default menu
state.apply_action(1)            # opponent calls
for card in (20, 25, 30):
    state.apply_action(card)
state.apply_street_raise_to(137)  # native raise-to is 275 + 137 = 412
view = state.decision_view()
assert view.pot == 687 and view.to_call == 137
inputs = HUNLFeatureEncoder().information_state(state)
assert inputs.shape == (540,)
```

## Limits and validation

OpenSpiel's bundled ACPC engine has a fixed action-history capacity per betting
round. In tests, a repeated-minimum-raise sequence is forced to call/fold around
64 actions in the round even when stacks remain. This is an engine limitation,
not a Hold'em rule or a deliberate learner abstraction. Do not claim complete
benchmark rule equivalence without checking that edge case. No engine fork is
introduced here. Its conservative whole-hand bound of 218 events is retained.

Tests compare exact transitions and payoffs against a separate native OpenSpiel
state over 80 seeded random hands, and check explicit chip examples, legal-size
deduplication, short all-ins, hidden-card isolation for both observer roles, suit
symmetry, board ordering, long histories, estimator support, cache equivalence,
checkpoint reload and exact sequential/parallel continuation. These tests verify
the adapter's consistency with OpenSpiel, not an independent proof of the
engine's correctness or playing strength.

The subsequent [API interface](API_INTERFACE.md) now implements benchmark clients
and public-state reconstruction with unknown opponent cards. Training samples
both private hands through OpenSpiel normally; live play uses the shared
`encode_decision` path and never calls the training critic.

Source: [OpenSpiel 1.6.3 universal poker implementation](https://github.com/google-deepmind/open_spiel/blob/v1.6.3/open_spiel/games/universal_poker/universal_poker.cc).
