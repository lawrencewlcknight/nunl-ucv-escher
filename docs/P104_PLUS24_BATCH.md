# HUNL P104 plus 24 active hours

One independent Google Cloud Batch task continues P104 to P128 and then
evaluates the five checkpoint matchups. It uses no chat automation or external
opponents. Preparation does not submit a paid job.

## Starting point and unchanged algorithm

- Source: `gs://clever-overview-399515-fhp-escher-results/hunl/hunl-p80-plus24-batch-20261120`.
- P104 iteration: **5359**, full recovery `training/state_005359`.
- Policy SHA-256: `b717b47b9afa0a8e1c2efc6214fb48d46d8cb069b85e873465fbf695a93c8307`.
- Trainer fingerprint: `b4510da50376a69972b2420cd7f5f2d7193291d61fc1624b457b817298b1704b`.
- Training archive SHA-256: `323c92b4e35dac18205abee34d78b445c1ad3552e70758b9dd49bff5898a26c1`.

The source already implements the corrected `objective_mass` average-policy
group sampler. Use that exact archived trainer without further migration,
metadata rebinding or baseline policy refitting. Resume networks, optimiser
states, replay, counters and worker random states from the full recovery.
Keep the architecture, learning budgets, game interface and encoder unchanged.

Submission validates published source metadata. The worker also verifies the
downloaded policy, archive and trainer fingerprint, then checksum-verifies and
fully reloads the recovery before training. Each exported policy must attest
to `objective_mass` and sampling seed `[20261115, outer_iteration]`.

## Checkpoints and evaluation

| Additional active hours | Policy | Adjacent comparison |
|---:|---|---|
| 6 | P110 | P110 vs P104 |
| 12 | P116 | P116 vs P110 |
| 18 | P122 | P122 vs P116 |
| 24 | P128 | P128 vs P122 |

Primary: **P128 vs P104**, 95% confidence interval. Four adjacent comparisons
use 99% intervals; secondary improvement claims require the primary test to
pass. Evaluation runs after training with eight workers and matched deals
with seats exchanged. Each match excludes a 10,000-pair pilot, whose variance
sets a fixed 100,000–1,000,000-pair main sample for estimated 80% power at
10 bb/100. Capped or incomplete samples are explicitly identified. There is
no Slumbot or GTO Wizard evaluation. One training seed and comparisons against
earlier policies do not establish exploitability or external playing strength.

Targets start from P104's recorded 374613.3848528917 active seconds and stop
at complete iteration boundaries. Average-policy extraction, saving, transfers
and evaluation are outside the active training budget. Every six-hour stage
exports a playable policy and reload-verifies a full recovery. Keep all five
playable policies and the newest verified full recovery; preserve earlier runs.
The six-hour wall-clock recovery schedule and recent-save suppression remain
unchanged. Checkpoint publication and verified retention use the existing path.

## Resources

- Project `clever-overview-399515`, region `europe-west1`.
- Standard `c4-highmem-16`, 200 GB Hyperdisk Balanced, eight collection workers
  and eight Torch threads.
- Same pinned runtime as P80→P104: Python 3.11.2, Torch 2.7.0+cpu,
  NumPy 1.26.4, OpenSpiel 1.6.3 and Ray 2.51.2.
- 30-hour training orchestration budget, four-hour evaluation budget,
  35-hour worker timeout, 36-hour Batch task limit and zero automatic retries.
- Batch releases its VM/disk after completion. Reports, source provenance,
  policies and the latest full recovery remain in cloud storage.

## Prepare and submit

```bash
cd /Users/lawrenceknight/Documents/deep_cfr_v3/hunl-ucv-escher
RUN="hunl-p104-plus24-batch-$(date -u +%Y%m%d-%H%M%S)"
python3 tools/launch_continuation_batch.py \
  --baseline-hours 104 --run-name "$RUN" --output "outputs/cloud-$RUN"
```

Submit that exact prepared bundle when ready:

```bash
python3 tools/launch_continuation_batch.py \
  --submit-prepared "outputs/cloud-$RUN"
```

Duplicate job names are rejected and a new cloud prefix is reserved before
submission. Inspect the saved submission record and Batch status before
retrying an interrupted submission.

## Download analysis only

Set `RUN` to the submitted job name. Copy only these three small reports;
do not recursively download the experiment prefix.

```bash
mkdir -p "outputs/$RUN-results/evaluation"
gcloud storage cp \
  "gs://clever-overview-399515-fhp-escher-results/hunl/$RUN/evaluation/continuation_report.md" \
  "gs://clever-overview-399515-fhp-escher-results/hunl/$RUN/evaluation/report.json" \
  "gs://clever-overview-399515-fhp-escher-results/hunl/$RUN/evaluation/design.json" \
  "outputs/$RUN-results/evaluation/" \
  --project=clever-overview-399515
```

The combined report contains all five comparisons, sample counts, confidence
intervals and completion flags. No weights, replay, checkpoints, archives or
raw hand records are included by this command.

## Validation and prepared bundle

The focused source-identity and Batch suite passed **43 tests**, including
bounded real continuation through P110/P116/P122/P128, full recovery reloads,
all five smoke evaluation matchups, rejected source mismatches and preservation
of the earlier P56/P80 launch paths. Production P104 metadata was downloaded
and validated; full production state download and reload occur on Batch before
training, not on the local laptop.

A local launch bundle is prepared at
`outputs/cloud-hunl-p104-plus24-batch-20261122/` for job
`hunl-p104-plus24-batch-20261122`. Its source checksum and shell syntax were
verified. It has not been submitted. To submit this existing bundle, use:

```bash
python3 tools/launch_continuation_batch.py \
  --submit-prepared outputs/cloud-hunl-p104-plus24-batch-20261122
```
