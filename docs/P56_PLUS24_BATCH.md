# HUNL P56 plus 24 active hours

One Google Cloud Batch task restores the completed P56 recovery, applies the
objective-mass average-policy sampler, trains for 24 additional active hours,
then evaluates its saved policies. No Slumbot, GTO Wizard, scheduled chat task,
or external evaluator is started. Batch releases its worker VM and disk when
the task finishes; artifacts persist in Cloud Storage.

## Source and migration

Source run: `hunl-p32-plus24-20261117-121800`, under
`gs://clever-overview-399515-fhp-escher-results/hunl`.
P56 is iteration **2941**, full recovery `training/state_002941`.
Its policy SHA-256 is
`a1ec410adc7aab98c07045a78ea37b42e7a4c7c3a6bb648afa7545a18d4ce94a`.

Before training, the worker checks the old archive and policy hashes and fully
restores the original state with its original source/runtime. It patches only
`unbiased_escher/holdem_fitting.py` and adds `unbiased_escher/policy_sampling.py`
inside a separate copy of the archived trainer. The old fitter's checksum and
exact source replacements are guarded. Current development trainer changes are
not substituted for the archive.

For grouped losses with weights `w`, sample independently with replacement
with probability `w / sum(w)`, multiplying each sampled loss by the training
partition's mean group weight. This preserves `sum(w * loss) / number_of_groups`
in expectation without applying the group weight twice. Full batches retain
the exact original weighted objective. Seeds are `[training_seed, iteration]`.
Validation membership, validation loss, fitting budgets, reset-each-fit
semantics and checkpoint selection are unchanged.

Migration changes only the recovery's source-identity metadata. Models,
optimisers, replay, counters, actor states and RNGs are preserved. A second
strict full restore with the patched archive must pass before training starts.
`sampler-migration.json` records the old/new fingerprints and changed files.
The original P56 policy is retained unchanged for comparison; there is no
zero-time refit of that baseline. Prior cloud runs are never overwritten.

## Training and comparison plan

| Additional active hours | Policy | Adjacent comparison |
|---:|---|---|
| 6 | P62 | P62 vs P56 |
| 12 | P68 | P68 vs P62 |
| 18 | P74 | P74 vs P68 |
| 24 | P80 | P80 vs P74 |

Active time excludes average-policy fitting, persistence, transfers and
evaluation, as in the preceding run. Targets are cumulative from P56's actual
recorded active time and stop at complete iteration boundaries.

All four stage recoveries are reloaded in fresh processes and policy weights
checked before publication. Preserve all five playable policies and the newest
full recovery in the new prefix; the preceding P56 recovery remains intact in
its original prefix. The archived trainer retains six-hour wall-clock recovery
checks with the existing recent-save suppression. Stage-boundary recovery
publication is synchronous; a failed task may lose work since its last
successfully published recovery. It never restarts training automatically.

After training, evaluate **P80 vs original P56** as the primary comparison
(95% interval), plus the four adjacent comparisons (99% intervals each).
Use exchanged seats on identical predealt cards, an excluded 10,000-pair pilot
per comparison, then a frozen 100,000–1,000,000-pair confirmation target based
on 80% estimated power for 10 bb/100. The four-hour evaluation cap may leave
samples incomplete; reports retain achieved counts and do not assert
significance for incomplete samples. Secondary improvement claims require
the primary comparison to pass. Evaluation uses eight workers; the primary
initially receives 40% of the budget.

This measures the combined effect of another 24 hours and the sampler fix for
one training seed. It does not isolate the fix, measure exploitability, or
guarantee monotonic improvement.

## Resources and execution

- One standard `c4-highmem-16`, 200 GB Hyperdisk Balanced boot disk.
- Eight collection workers, eight Torch threads, original networks and budgets.
- Pinned Debian container and Batch VM image; Python 3.11.2, Torch 2.7.0+cpu,
  NumPy 1.26.4, OpenSpiel 1.6.3 and Ray 2.51.2 match the P56 contract.
- 30-hour training orchestration budget, four-hour evaluation budget, 35-hour
  worker timeout and 36-hour Batch task maximum; zero automatic retries.
- Reports upload every two minutes; full recoveries publish at stage boundaries.
  Bootstrap logs and exit status upload on exit when the container can unwind.

Prepare a reviewable bundle (no cloud submission without `--execute`):

```bash
cd /Users/lawrenceknight/Documents/deep_cfr_v3/hunl-ucv-escher
RUN="hunl-p56-plus24-batch-$(date -u +%Y%m%d-%H%M%S)"
python3 tools/launch_continuation_batch.py \
  --run-name "$RUN" --output "outputs/cloud-$RUN" --execute
```

The launcher checks for duplicate Batch jobs, validates source metadata and
reserves a new storage prefix using generation-guarded writes. If submission
is interrupted, inspect the saved `launch.json`, `submission.json` and Batch
job before retrying; never blindly launch another paid task.

Key outputs are `batch_status.json`, `continuation.json`,
`sampler-migration.json`, `reload-P*.json`, `evaluation/report.json`,
`evaluation/continuation_report.md`, playable `P*.pkl` policies, and
`training/latest.json` with its full state shards. Future resumes must use
the **new** `training_source.tar.gz`, the saved contract and exact runtime.

Check the job and download analytical reports without weights or replay:

```bash
gcloud batch jobs describe "$RUN" --project=clever-overview-399515 \
  --location=europe-west1 --format='value(status.state)'
gcloud storage rsync --recursive \
  --exclude='(^|/)state_[0-9]+(/|$)|.*\.(pkl|pt|npy|npz|tar\.gz)$' \
  "gs://clever-overview-399515-fhp-escher-results/hunl/$RUN" \
  "outputs/$RUN-results"
```

## Validation and submitted run

The targeted test suite passed 40 tests, including a real one-worker smoke
recovery migration through all four stages and five local comparisons. It
compares every saved learning-state value except the intentionally changed
source identity, checks all stage reloads, and tests unbiased loss/gradient
scaling, failure reporting and duplicate submission prevention. The actual
production archive fingerprint and two-file patch were also checked locally;
the full eight-worker production reload runs on Batch before training starts.

Prepared run: `hunl-p56-plus24-batch-20261119-050144`. Its reviewable local
bundle is `outputs/cloud-hunl-p56-plus24-batch-20261119-050144/`. Submission
confirmation, when successful, is saved as `submission.json` in that bundle.
