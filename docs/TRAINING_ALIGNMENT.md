# Shared Hold'em training specification — v0.8.0

Both repositories use the same UCV estimator and fitting mechanics. Configuration
values may differ only for game rules, representation, network capacity justified
by no-limit betting complexity, and fresh regret-buffer capacity justified by
maximum trajectory length. A cross-repository regression test enforces that
allowlist when the sibling checkout is available.

| Setting | Both variants |
|---|---|
| Collection | 10,000 traversals per player per iteration |
| Regret fitting | 750 updates per player; frozen target cache; gradient norm cap 10 |
| Critic fitting | 250–2,500 updates per member, scaled by occupied replay |
| Calibration fitting | 100–2,000 updates, scaled by occupied replay |
| Minibatches | 2,048 for all learners |
| Learning rates | 0.001 regret/critic/calibration; 0.003 policy |
| Critics | Two cross-fit members; four-fit target averaging; cached targets |
| Control replay | 1 million critic rows total; 1 million calibration rows |
| Historical policy replay | 2 million rows |
| Exploration / estimator | 20% uniform floor; fixed beta 1; predictor disabled |
| Weighting | Alpha 2.3; gamma 2 |

For each control learner with occupied rows N and capacity C, the update count
is `min(maximum, max(minimum, ceil(maximum * N / C)))`. Skip fits until a full
minibatch exists. Each critic uses its own 500,000-row capacity. Skipped fits do
not advance target versions, optimizer state or temporal averaging history.

Both variants cache frozen regret targets once per fit, clip regret gradient
norms at 10, and skip empty regret batches without modifying learner state.
Collection and critic bootstrapping use uniform probabilities among tied maximum
regrets when all legal regrets are non-positive; unequal negative regrets still
select the highest value. Tiny positive regrets are normalized without a floor
that would reduce total probability. Terminal masks produce zero continuation.

The historical policy reservoir holds two million rows. Policy fitting resets
the model, runs all 20,000 updates, and selects the best iteration-weighted
held-out cross-entropy at 2,500/5,000/10,000/20,000 updates. A stable, game-neutral
hash holds out 5% of canonical feature groups; identical features cannot appear
in both partitions. The selected model and matching Adam state are restored.
Tiny replays without both partitions use all groups and report validation
unavailable. This selection is within one fit, not extra self-play experiments;
held-out policy error is not exploitability.

## Recovery and policy extraction

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

Only the newest recovery checkpoint is retained. The runner first writes and
verifies the new save, then atomically updates `latest.json`, then deletes all
older recognized recovery directories in that run. A failed save or pointer
update leaves the previous recovery available. Playable policy files, metrics,
unrelated directories and symlinks are not pruned. Saving temporarily requires
space for both checkpoints; cleanup time is included in `persistence_seconds`.
Metrics record the names removed in `recovery_checkpoints_deleted`.

## Deliberate differences

| Setting | No-limit | Fixed-limit |
|---|---|---|
| Actions | Seven-action menu | Three native fixed-limit actions |
| Regret/policy branches | 128 | 96 |
| Regret/policy trunk | 256 × 3 | 256 × 2 |
| Critic branches / trunk | 96 / 192 × 2 | Same |
| Calibration branches / trunk | 64 / 128 × 2 | Same |
| Fresh regret rows per player | 1 million | 250,000 |

At 10,000 limit traversals, 23 maximum decisions per hand bound fresh collection
at 230,000 rows. Reassess the limit buffer if increasing traversal counts.
No-limit's larger policy reservoir adds 2.336 GB of dense replay capacity over
its former one-million-row setting; grouping, fitting copies, workers and other
buffers require additional memory. No network dimensions or game encoders are
changed by this alignment.

Start a fresh training run. Strict source/configuration fingerprints reject
continuation from previous versions; saved policies remain playable. Validation
uses bounded engineering tests, not production training or configuration sweeps.
