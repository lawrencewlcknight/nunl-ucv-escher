# HUNL P200 against Slumbot

One independent Google Cloud Batch job scores the saved P200 average policy
against Slumbot. It can run alongside P200-to-P224 training: neither job modifies
the original P200 run or depends on the other job completing. There is no chat
automation, GTO Wizard evaluation, refitting or training in this benchmark.

| Setting | Value |
|---|---|
| Job | `hunl-p200-slumbot-20261127` |
| Scored sample / total fresh-hand cap | 8,000 hands |
| Pilot / separate live smoke | None |
| Score | Raw bb/100, two-sided 95% confidence interval |
| Action RNG seed | 2007901 |
| Session | Anonymous, private; no leaderboard login |
| Worker | `c4-standard-2`, one Torch thread |
| Task memory / boot disk | 6 GiB / 30 GB |
| Benchmark / worker / Batch task limits | 24 / 25 / 26 hours |
| Automatic task retries | Zero |

The 8,000-hand ceiling is the requested experimental budget, not a verified
Slumbot daily quota. Each hand can require multiple API calls. The previous P128
Slumbot run used 20,000 hands; this smaller sample will generally have a wider
confidence interval. Statistical significance is not guaranteed. The result
measures performance against Slumbot, not exploitability.

## Verified policy and failure handling

Source: `gs://clever-overview-399515-fhp-escher-results/hunl/hunl-p176-plus24-batch-20261126`.

- P200 iteration: **9844**.
- Policy SHA-256: `6a5b67a5f3da22b1d28c47ed2d98de27389c667aad94cd97dc47c3af6af7435b`.
- Completed continuation manifest SHA-256: `55b08d9fe715a671e5bf30576bcb87a22a7ef67e9fcf32861778a90647ec81b2`.
- Corrected average-policy sampler: `objective_mass`, seed `[20261115, 9844]`.

The worker downloads only `continuation.json`, `P200.json` and `P200.pkl` into
private input storage. Before play, it verifies the pinned hashes, completed
training/reload status, iteration and corrected grouping. It uses the policy's
saved encoder and action menu. Production model files are not downloaded locally
by experiment preparation or included in benchmark output.

This reuses the P128 benchmark implementation with an explicit `--policy-hours
200` selection. It retains the lessons from P32/P56 API failures: a single client,
180-second request timeout, minimum 0.5-second request spacing, and no blind retry
of an ambiguous Slumbot POST. A fresh-hand slot is reserved before the request;
an unresolved hand stops the benchmark rather than being replaced. There is no
automatic restart or extra pilot play. Unknown Slumbot version remains labelled
unverified.

Completed hands are journalled durably. Reports refresh every 25 hands and on
orderly failure; immutable snapshots publish every two minutes, followed by final
publication and a Batch cleanup step. Incomplete samples and unresolved attempts
are reported explicitly. Their intervals do not eliminate possible bias from
missing outcomes. Batch releases worker resources; Cloud Storage retains results.

## Prepare and submit

Prepare locally without launching paid work:

```bash
cd /Users/lawrenceknight/Documents/deep_cfr_v3/hunl-ucv-escher
python3 tools/launch_p128_benchmark.py \
  --policy-hours 200 --provider slumbot \
  --run-name hunl-p200-slumbot-20261127 \
  --output outputs/cloud-hunl-p200-slumbot-20261127
```

The launcher retains its existing filename for compatibility. Submit the prepared
bundle once:

```bash
python3 tools/launch_p128_benchmark.py \
  --submit-prepared outputs/cloud-hunl-p200-slumbot-20261127
```

Submission rechecks the source manifest, local bundle hashes and hand cap, rejects
an existing Batch job name, and reserves the destination prefix before upload.
If submission is interrupted, inspect the named job and its launch record before
retrying. Use a fresh name only for an intentionally separate experiment.

## Download analysis only

These four small files describe the sample, score, intervals and completion state.
They exclude models, source archives, startup logs and detailed hand journals:

```bash
cd /Users/lawrenceknight/Documents/deep_cfr_v3/hunl-ucv-escher
RUN="hunl-p200-slumbot-20261127"
PREFIX="gs://clever-overview-399515-fhp-escher-results/hunl/$RUN"
DEST="outputs/$RUN-results"
mkdir -p "$DEST/benchmark"
gcloud storage cp "$PREFIX/launch.json" "$PREFIX/cloud_status.json" "$DEST/" \
  --project=clever-overview-399515
gcloud storage cp "$PREFIX/benchmark/design.json" "$PREFIX/benchmark/report.json" \
  "$DEST/benchmark/" --project=clever-overview-399515
```

An early setup failure may leave no benchmark report; its absence is not a
completed evaluation. Offline tests cover the P200 selection, zero-pilot budget,
wire protocol, checksum rejection, duplicate-job prevention and publication.
No live poker API calls are used during preparation or testing.
