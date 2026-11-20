# Separate HUNL P56 and P80 strategic-position audits

Two independent Google Cloud Batch experiments audit the exported P56 and P80
policies. Each also re-scores the original P32 policy in the same runtime,
producing a matched P32-to-target comparison. Neither job trains, resumes
training, contacts external opponents or depends on the P80-to-P104 run.

## Frozen protocol

Use the exact library from the completed 18 November P32 audit, now checked into
`hunl_escher/data/strategic_positions_v1.json`. Its canonical JSON SHA-256 is
`3575612cbdcb672fc7d877d93fe041a7e8629134c661684a50012f5a5763dfc8`.
The launcher and worker reject changes to this library; they never regenerate it.

Only **374 development positions** are scored for each policy: 360 river
positions, 12 descriptive earlier-street probes, and two exact turn all-in
probes. The **152 assessment positions stay locked**. The existing audit
semantics, three synthetic river opponent panels, conditional ranges,
family-balanced aggregation and card-order/suit checks remain unchanged.

Each report contains action probabilities, exact conditional river values,
best-action gaps, guaranteed error costs, category breakdowns and matched
changes against P32. Earlier-street probes without an exact evaluator describe
behaviour without assigning guessed optimal actions. The turn all-in cases
use exact uniform-range equity. Costs are **bb per audited decision**, not
bb/100 hands or exploitability. These fixed, correlated positions do not support
population confidence intervals or prove whole-game improvement. Best-action
gaps and deployed conditional values are both reported because a candidate's
continuation policy can change alongside its current action probabilities.

The two target policies can also be compared directly after download: their
position IDs, library, opponent panels and evaluator are identical. P80 uses
the corrected group sampler; P56 predates that fix, so differences combine
further training and the correction.

## Pinned input policies

All sources are under `gs://clever-overview-399515-fhp-escher-results/hunl`.

| Policy | Source run | Iteration |
|---|---|---:|
| P32 reference | `hunl-plus24-20261116-074857` | 1658 |
| P56 target | `hunl-p32-plus24-20261117-121800` | 2941 |
| P80 target | `hunl-p56-plus24-batch-20261119-050144` | 4160 |

Policy hashes are pinned in `position_audit_config.py` and
`continuation_sources.py`, and recorded in each launch manifest. Submission
checks the publication markers. Workers download only each playable `.pkl`
and its `.json` marker, verify the bytes before loading, and check iteration
and training-time identity. Full training states are neither fetched nor altered.

## Resources, budgets and outputs

Each experiment uses one standard **c4-standard-2**, a 30 GB boot disk and one
Torch thread. It uses the existing pinned Batch container/runtime with CPU
PyTorch; Ray is not installed for these audits. The audit has a 600-second
cooperative work budget, the worker a 20-minute process guard, and the complete
Batch task a one-hour maximum including installation. Automatic retries are
disabled. A normal audit is expected to be much shorter than these limits;
setup and downloads may dominate runtime.

Analytical outputs are published on completion, failure or graceful interruption.
Partial coverage is explicitly marked incomplete and exits unsuccessfully.
Final COMPLETE status is published only after the result upload succeeds.
Abrupt host loss or hard termination can prevent the final upload. Batch
releases its worker VM and disk; results remain in Cloud Storage.

Outputs include `batch_status.json`, `policies.json`, `library.json`,
`audit/manifest.json`, `audit/report.json`, **`audit/report.html`**, all
`audit/records/*.json`, and bootstrap logs. Weights remain in the original
source runs and are not copied into the analytical result prefix.

## Prepare or launch

The two prepared local bundles are:

- `outputs/hunl-positions-p56-20261121-launch/`
- `outputs/hunl-positions-p80-20261121-launch/`

Submit them independently:

```bash
cd /Users/lawrenceknight/Documents/deep_cfr_v3/hunl-ucv-escher
python3 tools/launch_position_audit.py \
  --submit-prepared outputs/hunl-positions-p56-20261121-launch
python3 tools/launch_position_audit.py \
  --submit-prepared outputs/hunl-positions-p80-20261121-launch
```

For a fresh bundle, choose a new name and output directory. Preparation alone
does not contact GCP; add `--execute` to prepare and submit together:

```bash
RUN="hunl-positions-p56-$(date -u +%Y%m%d-%H%M%S)"
python3 tools/launch_position_audit.py --checkpoint-hours 56 \
  --run-name "$RUN" --output "outputs/$RUN-launch"
```

Use `--checkpoint-hours 80` for P80. Duplicate jobs and existing output prefixes
are rejected. Inspect any interrupted submission before retrying it.

Download either experiment's analytical results after completion:

```bash
RUN=hunl-positions-p56-20261121
gcloud batch jobs describe "$RUN" --project=clever-overview-399515 \
  --location=europe-west1 --format='value(status.state)'
gcloud storage rsync --recursive --exclude='.*\.(pkl|pt|npy|npz|tar\.gz)$' \
  "gs://clever-overview-399515-fhp-escher-results/hunl/$RUN" \
  "outputs/$RUN-results"
```

Repeat with `RUN=hunl-positions-p80-20261121` for the other job. No scheduled
chat task is required.

## Validation

The focused audit, Batch contract and source-identity suite passed **42 tests**.
This covers both target policies, locked assessment, partial/failure publication,
bad policy bytes and metadata, duplicate/network rejection before submission,
bundle contents, shell syntax and unchanged continuation bootstrap defaults.
A real P32 rerun completed all 374 development positions in 7.47 seconds locally
and reproduced every historical river-panel summary field within 1e-8.
The two Batch bundles are prepared locally and have not been submitted.
