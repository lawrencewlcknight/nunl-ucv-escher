# HUNL P224 plus 72 active hours

One autonomous Google Cloud Batch job continues the verified P224 full learning
state to P296. It preserves the corrected average-policy grouping, archived
trainer, network sizes, replay capacities, encoder, action abstraction, training
schedule and random state. There are no Slumbot/GTO calls or chat automations.
Preparation alone does not launch paid work.

## Verified starting point

- Source: `gs://clever-overview-399515-fhp-escher-results/hunl/hunl-p200-plus24-batch-20261127`.
- P224 iteration **10913**, full recovery `training/state_010913`.
- Policy SHA-256: `3d1b62241deacbebdbe0ccfa65bf6410a769a22ec2d697274955401df9fba6fa`.
- Trainer fingerprint: `b4510da50376a69972b2420cd7f5f2d7193291d61fc1624b457b817298b1704b`.
- Archived trainer SHA-256: `323c92b4e35dac18205abee34d78b445c1ad3552e70758b9dd49bff5898a26c1`.
- Recorded active training: **806818.7288267743 seconds**.
- Average-policy grouping: `objective_mass`, seed `[20261115, 10913]`.

Published source metadata has been checked. Submission rechecks it, and the Batch
worker verifies archive/policy checksums, checks every recovery shard and performs
a full state reload before training. The original P224 run remains intact.
The continuation resumes optimisers, replay, counters and worker random states;
there is no fresh training, metadata migration or baseline-policy refit.

## Checkpoints and evaluation

| Additional active hours | Checkpoint | Adjacent comparison |
|---:|---|---|
| 6 | P230 | P230 vs P224 |
| 12 | P236 | P236 vs P230 |
| 18 | P242 | P242 vs P236 |
| 24 | P248 | P248 vs P242 |
| 30 | P254 | P254 vs P248 |
| 36 | P260 | P260 vs P254 |
| 42 | P266 | P266 vs P260 |
| 48 | P272 | P272 vs P266 |
| 54 | P278 | P278 vs P272 |
| 60 | P284 | P284 vs P278 |
| 66 | P290 | P290 vs P284 |
| 72 | P296 | P296 vs P290 |

Use cumulative targets relative to P224's actual active time, finishing whole
iterations without accumulating twelve stage overshoots. Export each policy and
save, reload-verify and publish its recovery before proceeding. Retain all 13
playable policies and the newest verified full recovery; preserve earlier runs.
The existing six-hour wall-clock recovery schedule, with suppression when a
checkpoint was saved in the previous hour, is unchanged.

After training, eight evaluation workers run **13 comparisons**: the primary
**P296 vs P224** at 95% confidence, then all twelve adjacent comparisons above.
The secondary alpha is **0.04/12 = 0.003333…**, giving approximately **99.667%**
intervals. This retains the previous conservative 4% secondary-family error
budget across the larger family. Secondary improvement claims also require the
primary test to pass. Older 24-hour experiments retain their four 99% comparisons.

Use matched deals with seats exchanged, analysing each two-hand pair as one
observation. Each match retains an excluded 10,000-pair variance pilot; select a
fixed main sample of 100,000–1,000,000 pairs for 80% power at a 10 bb/100 effect.
The stricter secondary alpha can require larger samples. Runtime/sample caps,
failures and incomplete samples remain explicit. These are comparisons within
one training lineage, not exploitability or external-opponent measurements.

## Runtime and resources

- Project `clever-overview-399515`, region `europe-west1`.
- Standard `c4-highmem-16`, 200 GB Hyperdisk Balanced.
- Eight collection workers and eight Torch threads, unchanged.
- Same pinned checkpoint-compatible runtime and archived trainer as P224.
- **72 active training hours**, excluding checkpoint extraction, saving, transfers
  and evaluation; **90-hour training orchestration limit**.
- **12-hour evaluation limit**, scaled for the larger comparison family.
- **103-hour worker timeout; 104-hour Batch task hard stop**.
- Zero automatic retries. A failure does not silently restart the experiment.

The last 24-hour continuation needed approximately 24.76 wall hours before
evaluation. At similar throughput, allow roughly **76–80 hours overall**. The
larger time limits are ceilings, not intentional waiting or reserved compute;
Batch releases workers when the task finishes. Results and resumable recovery
remain in Cloud Storage. A later continuation can resume the final verified P296
state with its archived trainer.

## Prepare and submit

Prepare a bundle locally:

```bash
cd /Users/lawrenceknight/Documents/deep_cfr_v3/hunl-ucv-escher
python3 tools/launch_continuation_batch.py \
  --baseline-hours 224 --additional-hours 72 \
  --run-name hunl-p224-plus72-batch-20261128 \
  --output outputs/cloud-hunl-p224-plus72-batch-20261128
```

The bundle is prepared locally during implementation. Submit that bundle once:

```bash
python3 tools/launch_continuation_batch.py \
  --submit-prepared outputs/cloud-hunl-p224-plus72-batch-20261128
```

Preparation will refuse an existing local directory. Submission checks the source
archive and production metadata, rejects an existing job name and reserves the
cloud prefix before upload. If submission is interrupted, inspect the named job
and saved submission record before retrying. Do not create a duplicate run.

## Download analysis only

```bash
cd /Users/lawrenceknight/Documents/deep_cfr_v3/hunl-ucv-escher
RUN="hunl-p224-plus72-batch-20261128"
PREFIX="gs://clever-overview-399515-fhp-escher-results/hunl/$RUN"
DEST="outputs/$RUN-results"
mkdir -p "$DEST/evaluation"
gcloud storage cp "$PREFIX/continuation.json" "$PREFIX/batch_status.json" \
  "$DEST/" --project=clever-overview-399515
gcloud storage cp "$PREFIX/evaluation/continuation_report.md" \
  "$PREFIX/evaluation/report.json" "$PREFIX/evaluation/design.json" \
  "$DEST/evaluation/" --project=clever-overview-399515
```

These five files include training progress, all 13 comparisons, confidence
intervals, sample sizes and completion flags. No models, recovery states,
source archives or logs are downloaded.

## Validation

158 selected tests passed, including a bounded real twelve-stage P224-to-P296
continuation, reloads of all checkpoint states, all thirteen offline comparisons,
the expanded secondary error control, historical 24-hour continuation behaviour,
source identity rejection and Batch success/failure publication. Seven older
full migration smoke cases were not repeated.

The prepared archive matches the local source and its Batch shell passes syntax
validation. Production provenance was checked using six small cloud metadata
files; the production recovery itself is checksum-verified and fully reloaded on
Batch before training. No production model was downloaded locally and no job was
submitted during preparation.
