# HUNL network configuration

The default allocates more capacity to regret and average-policy prediction,
while keeping critics and residual calibration compact to control repeated
fitting cost. This is a considered starting point, not an experimentally proven
optimum or a demonstrated improvement in playing strength.

| Role | Models | Width of each input branch | Joint hidden layers | Parameters per model |
|---|---:|---:|---|---:|
| Cumulative regret | 2 | 128 | 256 → 256 → 256 | 305,159 |
| Average policy | 1 | 128 | 256 → 256 → 256 | 305,159 |
| Cross-fit critic | 2 | 96 | 192 → 192 | 160,231 |
| Residual calibration | 1 | 64 | 128 → 128 | 78,722 |

There are three branches: exact cards, strategic cards, and betting context.
Each branch is one dense ReLU layer. Their outputs are concatenated and passed
through the listed joint ReLU layers, then the existing linear output layer.
Counts use the v5 encoder and default seven-action menu: policy/regret inputs
569, critic inputs 688, calibration inputs 579; seven outputs except the
two-output calibration model. Alternative action menus change these counts.

The six online models total **1,314,661 parameters**, versus 1,148,645 with the v3 encoder.
Their float32 weights occupy approximately **5.26 MB** (decimal). This excludes
target copies, temporal critic history, optimiser state, gradients, activations
and replay. Replay remains the much larger memory allocation. Larger models
also increase fitting cost; parameter memory alone does not determine VM needs.

## Configuration and training

`hunl_escher/data/baseline.json` contains independent `<role>_branch_width` and
`<role>_network_layers` fields, where role is `regret`, `average_policy`, `critic`
or `calibration`. Layer lists describe only the joint trunk. They may have
different depths and unequal widths. Branch widths and all hidden widths must
be positive integers. These fields replace the old shared `num_layers`,
`num_hiddens` and `structured_branch_width` settings.

The same configuration constructs both the learner and parallel traversal
workers. Smoke uses these full-sized networks with reduced replay and fitting
budgets. Production uses 750 regret steps per player, up to 2,500 critic steps
per member, up to 2,000 calibration steps per iteration and 20,000 policy steps
per extraction. Critic/calibration budgets scale with replay occupancy during
warm-up; full buffers recover the mature budgets. Frozen regret targets are cached once per fit; regret gradients are clipped
at norm 10. Policy replay holds two million rows and a 5% grouped holdout
selects the policy model within each 20,000-update fit. Learning rates, losses, widths and target averaging are retained.
See [the production schedule](BASELINE.md#production-schedule-and-monitoring).

Start a new output directory for training with these defaults. Continuation
requires identical configuration and source fingerprint, so an older run cannot
resume into the larger architecture. Policy checkpoints independently record
their actual branch widths and hidden layers: previous v3 policies remain
loadable for inference, including through the opponent adapters. This does not
restore compatibility with the rejected v2 feature encoder.

## Validation

Tests cover exact default parameter counts, distinct per-role custom shapes,
finite training updates, previous/current policy-shape reload, changed-config
continuation rejection, cached/uncached agreement and exact sequential and Ray
continuation. The offline opponent tests exercise the same saved-policy loader.
These are correctness checks; full-budget throughput and playing strength still
need measurement on the intended training machine.
