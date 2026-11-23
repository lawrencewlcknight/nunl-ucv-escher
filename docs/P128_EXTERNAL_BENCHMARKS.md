# HUNL P128 against Slumbot and GTO Wizard

Two independent Google Cloud Batch experiments evaluate the existing **trained
P128 average policy**. No training, refitting, replay loading, chat automation or
cross-job dependency is involved.

| Setting | Slumbot | GTO Wizard |
|---|---|---|
| Suggested job | `hunl-p128-slumbot-20261123` | `hunl-p128-gtow-20261123` |
| Excluded pilot | None | None |
| Scored sample | 20,000 hands | 8,000 hands |
| Total fresh-hand cap | 20,000 | **8,000, including all setup/pilot play** |
| Primary score | Raw bb/100 | AIVAT bb/100 |
| Secondary score | Same raw score | Raw bb/100 |
| Action RNG seed | 1287901 | 1288901 |
| Benchmark budget | 24 hours | 24 hours |
| Batch task hard stop | 26 hours | 26 hours |
| Worker | `c4-standard-2`, 2 vCPU, 1 Torch thread | Same |
| Task memory / disk | 6 GiB / 30 GB | Same |
| Automatic task retries | **0** | **0** |

Each job has a sequential API client and a separate immutable Cloud Storage
prefix. Batch manages worker resources. Either provider can fail without blocking
or invalidating the other experiment. GTO uses the existing Secret Manager secret
`gtowizard-benchmark-api-key`; Slumbot is anonymous and requires no secret.
The GTO 8,000-hand allowance is a per-experiment ceiling, not a reservation of shared
account quota. Other account activity can still cause an earlier quota refusal.

## Exact source policy

Source: `gs://clever-overview-399515-fhp-escher-results/hunl/hunl-p104-plus24-batch-20261122`.

- Policy: `P128.pkl`, iteration **6514**, SHA-256
  `aade5f09a22fc7aeddc7063e4250fc00f04c16cc5c4827381f775e3d66727bb3`.
- Completed training manifest SHA-256:
  `a08210f79bcddebc564109886673d38cb7c8002326b682b2b66f085eb54c048b`.
- Verified final recovery; the recorded policy fit uses `objective_mass` grouping.

The worker downloads only `continuation.json`, `P128.json` and `P128.pkl` into
private input storage. It checks pinned hashes, completed training/reload status,
marker agreement and checkpoint iteration **before any poker API call**. It loads
the saved policy and its own encoder/action menu. It never uploads model files as
benchmark results. The source archive is separately checksummed and includes the
current API adapter fixes.

## Lessons from the earlier external runs

The original P32 Slumbot benchmark stopped on an ambiguous POST timeout after
1,000 pilot and 8,044 scored hands. P56 stopped after 60 pilot hands. The previous
Slumbot transport used a 30-second timeout. This experiment raises it to **180
seconds**, keeps a minimum 0.5-second request interval, and allows only one client.
This mitigates slow responses; it cannot guarantee an uninterrupted 20,000 hands.

Slumbot POST mutations are **never blindly retried**. An unavailable action result
may conceal an applied action or a completed hand; replacing it with a new hand
could bias results. An ambiguous response stops the run, retaining every verified
completed score and identifying the unresolved attempt. It is never counted as a
zero or silently dropped into an apparently complete sample. There is no automatic
restart of a partially played experiment.

GTO retains the fixes from the earlier integration: the named research User-Agent,
180-second timeout, bounded documented 502/503/504 engine-busy retries, redacted
409 diagnostics without retries, terminal hands returned immediately by `/hands`,
and authoritative recovery only when an action's application can be proved.
The retry status list follows the [official client](https://github.com/gtowizard-ai/researcher-api-client/blob/main/src/utils.py).
Existing unfinished account hands are excluded; authentication checks use GETs and
consume no hands. Opponent version is checked before play, every 100 scored hands
and at the end; a version change invalidates the pooled benchmark.

There is **no live smoke test and no pilot for either opponent**. Every requested fresh hand reserves
one slot before the call. The fixed main loop plus an independent guard prevents a
8,001st logical new-hand request. Documented busy retries concern the same request,
not replacement hands. Ambiguous starts stop the job rather than freeing a slot.

Each action and completed hand is journalled with flush/fsync. Atomic reports are
updated every 25 completed hands and on orderly failure. Stable report/journal
snapshots are uploaded every two minutes, plus mandatory final publication and a
Batch `alwaysRun` publication step. Host loss can still lose the newest unpublished
prefix. Request deadlines also apply within a hand; a request already in progress
can consume its bounded timeout. Setup uses an exported PATH, isolated Python,
apt lock waiting and bounded source/download checks from the corrected GTO job.

## Statistical interpretation

Both samples are fixed before play, with every completed hand included in the
scored sample. Neither experiment uses a pilot. Reports include hand counts, raw/primary scores and two-sided 95%
intervals in bb/100. These are descriptive estimates of one policy against each
opponent, not exploitability measurements or guaranteed-powered improvement tests.
A fixed 8,000-hand sample does not guarantee statistical significance. AIVAT reduces
variance but does not make the GTO and Slumbot scores directly interchangeable.

Partial samples retain intervals but are explicitly incomplete: response failures
can cause non-random missing outcomes. Earlier partial P32/P56 scores are not pooled
into this experiment or presented as a controlled improvement comparison. Unknown
Slumbot version is reported as unverified.

## Prepare and submit separately

From the HUNL repository, these commands prepare bundles without contacting GCP:

```bash
cd /Users/lawrenceknight/Documents/deep_cfr_v3/hunl-ucv-escher

python3 tools/launch_p128_benchmark.py \
  --provider slumbot --run-name hunl-p128-slumbot-20261123 \
  --output outputs/cloud-hunl-p128-slumbot-20261123

python3 tools/launch_p128_benchmark.py \
  --provider gtow --run-name hunl-p128-gtow-20261123 \
  --output outputs/cloud-hunl-p128-gtow-20261123
```

Once prepared, submit each exact bundle once:

```bash
python3 tools/launch_p128_benchmark.py \
  --submit-prepared outputs/cloud-hunl-p128-slumbot-20261123

python3 tools/launch_p128_benchmark.py \
  --submit-prepared outputs/cloud-hunl-p128-gtow-20261123
```

Submission rechecks the local archive/job hashes and remote training manifest.
Generation-guarded uploads reserve the output prefix. If submission is interrupted,
inspect the named Batch job and its launch record before doing anything else;
do not rerun under another name and accidentally spend another 8,000 GTO hands.
For a different date use fresh matching names in the preparation commands. Do not
reuse an existing output directory or cloud prefix.

The service account needs Batch execution, source-bucket read, destination-bucket
write, and (GTO only) access to the named secret. No key is embedded in a source
archive, Batch specification, report, journal or startup command.

## Download analysis only

These explicitly selected files contain the experiment design, complete or partial
scores, confidence intervals and worker status. They exclude models, replay,
archives, startup logs and detailed hand journals:

```bash
cd /Users/lawrenceknight/Documents/deep_cfr_v3/hunl-ucv-escher
for opponent in slumbot gtow; do
  run="hunl-p128-${opponent}-20261123"
  destination="outputs/${run}-results"
  prefix="gs://clever-overview-399515-fhp-escher-results/hunl/${run}"
  mkdir -p "$destination/benchmark"
  gcloud storage cp "$prefix/launch.json" "$prefix/cloud_status.json" "$destination/"
  gcloud storage cp "$prefix/benchmark/design.json" "$prefix/benchmark/report.json" "$destination/benchmark/"
done
```

Neither experiment produces a pilot-precision report.
An early setup failure may leave only `cloud_status.json` or startup diagnostics;
absence of `report.json` must not be interpreted as a completed evaluation.
