# HUNL P80 plus 24 active hours

One self-contained Google Cloud Batch task resumes P80, trains for another
24 active hours, then evaluates five checkpoint matchups. No Slumbot, GTO
Wizard, chat automation or separately launched evaluator is involved.

## Verified starting point

- Source run: `hunl-p56-plus24-batch-20261119-050144` under
  `gs://clever-overview-399515-fhp-escher-results/hunl`.
- P80: iteration **4160**, recovery `training/state_004160`.
- Policy SHA-256:
  `a489504dada7fcb98501a6655c705f4ad679a46cd7979aad15bdbe63483acc5c`.
- Archived trainer fingerprint:
  `b4510da50376a69972b2420cd7f5f2d7193291d61fc1624b457b817298b1704b`.
- Training archive SHA-256:
  `323c92b4e35dac18205abee34d78b445c1ad3552e70758b9dd49bff5898a26c1`.

P80 already uses the corrected average-policy group sampler. Continue with
that exact archived trainer: **no further migration, metadata rebinding,
baseline refit or algorithm change**. Models, optimisers, replay, counters and
worker random states resume from the full recovery. Source metadata, archive
checksum and trainer fingerprint are checked, followed by a full solver and
actor reload with policy equality verification before training starts.

The existing sampler draws groups with replacement proportional to their
objective weights, with the mean group weight applied to sampled losses.
It preserves the weighted objective in expectation without double weighting.
Seeds remain `[training_seed, outer_iteration]`. Each new policy's diagnostics
must attest to `objective_mass` and the expected seed before evaluation.

## Training and evaluation

| Additional active hours | Export | Adjacent comparison |
|---:|---|---|
| 6 | P86 | P86 vs P80 |
| 12 | P92 | P92 vs P86 |
| 18 | P98 | P98 vs P92 |
| 24 | P104 | P104 vs P98 |

The primary comparison is **P104 vs P80**, with a 95% confidence interval.
The four adjacent comparisons use 99% intervals; secondary improvement claims
require the primary test to pass. Evaluation runs after training, using eight
workers, identical predealt cards with exchanged seats, and an excluded
10,000-pair pilot per comparison. Pilot variance sets a frozen confirmation
target of 100,000–1,000,000 pairs for estimated 80% power at 10 bb/100. A
four-hour evaluation cap is retained; incomplete samples are explicitly
reported and cannot produce confirmatory improvement claims.

Active training time excludes average-policy extraction, saving, transfers
and evaluation. Targets are cumulative from P80's actual recorded time,
with stopping at complete iteration boundaries. Every stage exports a policy
and fully reload-verifies its recovery before publication. Keep all five
playable policies and the newest full recovery; earlier runs remain intact.
The archived six-hour wall-clock recovery schedule and recent-save suppression
remain unchanged. Checkpoint publication is synchronous at stage boundaries.

This is one training seed and measures relative performance against earlier
policies. Both endpoints already use the corrected sampler. It does not
measure exploitability or strength against external opponents.

## Resources and safeguards

- Standard `c4-highmem-16`, 200 GB Hyperdisk Balanced disk.
- Eight collection workers and eight Torch threads; unchanged learning budgets.
- Same pinned container and Batch VM image as the P56-to-P80 experiment;
  Python 3.11.2, Torch 2.7.0+cpu, NumPy 1.26.4, OpenSpiel 1.6.3, Ray 2.51.2.
- 30-hour training orchestration budget, four-hour evaluation budget,
  35-hour worker timeout and 36-hour Batch task limit; zero automatic retries.
- Reports upload every two minutes; Batch releases the VM and disk after
  completion. Policies, full recovery, source archive and reports remain in GCS.

## Prepare and submit

Preparation is local and does not start a paid job:

```bash
cd /Users/lawrenceknight/Documents/deep_cfr_v3/hunl-ucv-escher
RUN="hunl-p80-plus24-batch-$(date -u +%Y%m%d-%H%M%S)"
python3 tools/launch_continuation_batch.py \
  --baseline-hours 80 --run-name "$RUN" --output "outputs/cloud-$RUN"
```

Submit that exact bundle:

```bash
python3 tools/launch_continuation_batch.py \
  --submit-prepared "outputs/cloud-$RUN"
```

Alternatively append `--execute` to preparation to prepare and submit together.
The launcher rejects duplicate jobs, checks cloud source metadata and reserves
an independent output prefix with generation-guarded writes. If submission
is interrupted, inspect the saved plan, submission record and Batch job before
retrying. Never blindly rerun an expensive experiment.

```bash
gcloud batch jobs describe "$RUN" --project=clever-overview-399515 \
  --location=europe-west1 --format='value(status.state)'
gcloud storage rsync --recursive \
  --exclude='(^|/)state_[0-9]+(/|$)|.*\.(pkl|pt|npy|npz|tar\.gz)$' \
  "gs://clever-overview-399515-fhp-escher-results/hunl/$RUN" \
  "outputs/$RUN-results"
```

Inspect `batch_status.json`, `continuation.json`, `source-verification.json`,
`reload-P*.json`, `evaluation/report.json` and
`evaluation/continuation_report.md`. `resume.json` records the exact archived
trainer required to resume from P104; playable policy weights alone cannot
resume training.

## Local validation

The focused continuation, Batch, source-identity and sampler suite passed
**52 tests**. This includes real bounded one-worker training through P86,
P92, P98 and P104, full reload verification at each boundary, and all five
local evaluation matchups. Negative tests reject wrong policy identities,
recovery pointers, trainer hashes, sampler attestations and production settings.
The P56 launch path remains covered. The actual eight-worker P80 production
restore is performed on Batch before training, using its pinned runtime.

A local bundle is prepared at
`outputs/cloud-hunl-p80-plus24-batch-20261120/`; it has not been submitted.
