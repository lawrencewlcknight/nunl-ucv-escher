# HUNL P152 plus 24 active hours

One independent Google Cloud Batch job resumes P152, exports P158/P164/P170/P176,
and then runs five local checkpoint comparisons. The job runs autonomously on
GCP with no chat automation and no Slumbot or GTO Wizard calls. Preparation does
not submit a paid job. Earlier external evaluations remain independent.

## Verified starting point

- Source: `gs://clever-overview-399515-fhp-escher-results/hunl/hunl-p128-plus24-batch-20261123`.
- P152 iteration: **7630**, full recovery `training/state_007630`.
- Policy SHA-256: `a0b0d5b5cde7a9d186be843650063bc7a5b4eb949c038686b253dacad3d94ca2`.
- Trainer fingerprint: `b4510da50376a69972b2420cd7f5f2d7193291d61fc1624b457b817298b1704b`.
- Archived trainer SHA-256: `323c92b4e35dac18205abee34d78b445c1ad3552e70758b9dd49bff5898a26c1`.

The source already uses the corrected `objective_mass` average-policy group
sampler. Resume the exact archived trainer and full saved state: networks,
optimisers, replay buffers, counters and worker random states. Do not rebind
metadata, migrate the trainer, refit P152 or change the network sizes, buffers,
encoder, action abstraction or training schedule.

Submission rechecks published source metadata. The worker verifies the policy,
archived source and trainer fingerprint, checksum-verifies the recovery, and
fully reloads it before any training. Each new policy must attest to the corrected
group sampler and sampling seed `[20261115, outer_iteration]`.

## Training and checkpoint schedule

| Additional active training | Policy | Adjacent comparison |
|---:|---|---|
| 6 hours | P158 | P158 vs P152 |
| 12 hours | P164 | P164 vs P158 |
| 18 hours | P170 | P170 vs P164 |
| 24 hours | P176 | P176 vs P170 |

Targets start from P152's recorded **547493.5978068254 active seconds**.
Use cumulative targets and finish complete iterations, avoiding accumulated
six-hour-stage overshoots. Policy extraction, persistence, transfers and evaluation
are outside the active training budget.

Each stage exports a playable policy and saves, reload-verifies and publishes a
full recovery through the existing checkpoint path. Keep all five playable
policies and the newest verified full recovery. Preserve the original P152 run
and all other previous runs. The six-hour wall-clock recovery schedule and
suppression when a checkpoint was saved within the previous hour stay unchanged.

## Evaluation after training

Primary comparison: **P176 vs P152**, with a **95% confidence interval**.
The four adjacent comparisons use **99% intervals**; secondary improvement claims
require the primary test to pass. Eight evaluation workers use matched deals with
seats exchanged, clustering each exchanged-seat pair as one observation.

Each comparison retains the established local evaluation's excluded 10,000-pair
variance pilot. Its variance determines a fixed 100,000–1,000,000-pair main sample,
targeting 80% power for a 10 bb/100 difference. This pilot is local simulation,
not an external-provider benchmark. Capped or incomplete samples are flagged.
These comparisons concern one training lineage and do not establish exploitability
or performance against an external opponent.

## Runtime and resources

- Project `clever-overview-399515`, region `europe-west1`.
- Standard `c4-highmem-16`, 200 GB Hyperdisk Balanced.
- **Eight collection workers and eight Torch threads**, unchanged from P152.
- Pinned runtime: Python 3.11.2, Torch 2.7.0+cpu, NumPy 1.26.4,
  OpenSpiel 1.6.3 and Ray 2.51.2.
- Training orchestration budget: **30 hours**; local evaluation budget: **4 hours**.
- Worker timeout: **35 hours**; Batch task hard stop: **36 hours**.
- **Zero automatic retries**. A failed job must not silently restart training.

Batch releases its worker resources after the job. Cloud Storage retains reports,
source provenance, playable policies and the latest full recovery, including
instructions for another continuation. Analytical downloads exclude model files.
The previous continuation's duration suggests roughly 26–28 hours overall, but
transfer time and evaluation sample sizes can change that estimate.

## Prepare and submit

Prepare a fresh bundle locally, without launching a job:

```bash
cd /Users/lawrenceknight/Documents/deep_cfr_v3/hunl-ucv-escher
RUN="hunl-p152-plus24-batch-$(date -u +%Y%m%d-%H%M%S)"
python3 tools/launch_continuation_batch.py \
  --baseline-hours 152 --run-name "$RUN" --output "outputs/cloud-$RUN"
```

Submit the exact prepared bundle:

```bash
python3 tools/launch_continuation_batch.py \
  --submit-prepared "outputs/cloud-$RUN"
```

A fixed-name bundle is also prepared locally at
`outputs/cloud-hunl-p152-plus24-batch-20261125/`. To submit that bundle instead:

```bash
python3 tools/launch_continuation_batch.py \
  --submit-prepared outputs/cloud-hunl-p152-plus24-batch-20261125
```

Choose one bundle, not both. Submission rejects duplicate Batch job names and
reserves the new Cloud Storage prefix before uploading. Inspect the named job and
saved submission record before retrying an interrupted submission.

## Download analysis only

Set `RUN` to the actual submitted job name. Download these small reports only:

```bash
cd /Users/lawrenceknight/Documents/deep_cfr_v3/hunl-ucv-escher
RUN="hunl-p152-plus24-batch-20261125"
mkdir -p "outputs/$RUN-results/evaluation"
gcloud storage cp \
  "gs://clever-overview-399515-fhp-escher-results/hunl/$RUN/continuation.json" \
  "gs://clever-overview-399515-fhp-escher-results/hunl/$RUN/batch_status.json" \
  "outputs/$RUN-results/" --project=clever-overview-399515
gcloud storage cp \
  "gs://clever-overview-399515-fhp-escher-results/hunl/$RUN/evaluation/continuation_report.md" \
  "gs://clever-overview-399515-fhp-escher-results/hunl/$RUN/evaluation/report.json" \
  "gs://clever-overview-399515-fhp-escher-results/hunl/$RUN/evaluation/design.json" \
  "outputs/$RUN-results/evaluation/" --project=clever-overview-399515
```

The reports contain training progress and all five comparisons, including sample
counts, intervals and completion flags. No recursive download, model weights,
replay, source archive or logs are needed for this analysis.

## Validation

All **81 focused tests passed**, covering bounded continuation, checkpoint reloads,
all five comparisons, incompatible source rejection and deployment limits for this
baseline and earlier baselines. Production P152 metadata was downloaded and validated;
the prepared archive checksum and shell syntax were checked. Source metadata is
rechecked before submission and the full recovery is checksum-verified and reloaded
on Batch before training. No production model files were downloaded locally.

This implementation prepares the experiment only; no Batch job was submitted.
