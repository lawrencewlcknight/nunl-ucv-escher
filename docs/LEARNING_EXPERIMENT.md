# Four-plus-four-hour learning check

This experiment tests one HUNL training seed at 200bb, blinds 50/100, no rake.
It implements an engineering and early-learning check, not an exploitability
measurement or a guarantee of statistically significant improvement.

## Training and artifacts

The production network, encoder, replay capacities and learning budgets are
unchanged. The orchestrator starts a fresh training subprocess for four hours of
active collection/network fitting, finishes its current iteration, fits the
average policy and saves full recovery state. It publishes `P4.pkl` and `P4.json`,
then starts a **new process with `--resume`** for four additional active hours.
It publishes `P8.pkl` and `P8.json` with actual times, iterations and hashes.
Both fits use the existing 20,000-update held-out policy selection procedure.
No exploration floor is applied at evaluation time; actions are sampled from
the saved average policy. Inference uses no critics or hidden opponent cards.

Policy extraction, persistence, restart, hardware profiling and evaluation do
not consume the eight active training hours. Iteration boundaries can overshoot
each four-hour target. Recovery remains on the six-hour wall grid, with the
one-hour recent-save exclusion; endpoint saves are unconditional. Latest-only
recovery retention is preserved, while both playable policies remain available.
An uploaded P8 recovery replaces the earlier cloud recovery only after the new
state and latest pointer have uploaded successfully.

The training wall budget is **12 hours**, including preflight and restart;
local evaluation has an additional **two-hour** budget. A budget overrun or
iteration ceiling is reported as failure/incomplete, not a successful 4+4 run.
Hard process deadlines may interrupt a fit; the last completed recovery and
policy remain available. The full production run has not been executed locally.

Install with Python 3.10–3.12:

```bash
python -m pip install -e '.[parallel,api,dev]'

# Offline end-to-end engineering run, using full network widths and tiny replay/update budgets.
# Choose a NEW output directory for each invocation.
python -m hunl_escher.experiment --smoke --stage-seconds 0.01 \
  --training-wall-seconds 180 --evaluation-seconds 180 --output outputs/learning-smoke

# Eight active hours on a suitable VM. No live opponents or cloud resources are created here.
python -m hunl_escher.experiment --output outputs/learning-4plus4
```

The smoke run produces eight confirmatory pairs per comparison. Its statistical
output is solely a plumbing check, never evidence of playing strength.

`train.py --additional-training-seconds N` is also independently usable with
`--resume`. N means additional active time for this process, not lifetime time.
Use a sufficiently large `--iterations` ceiling; either limit can stop training.
The active-time limit is not part of the immutable continuation contract.

## Primary statistical comparison

P8 plays P4 on duplicated, fully predealt hands. Both private hands and the board
stay assigned to the same seats; policies swap seats for the second hand. Chance
draws and stochastic action streams use separate deterministic streams. The
independent sampling unit is the **mean P8 profit across a two-hand pair**, in bb.
No private opponent cards are passed to policy inference.

1. Run 10,000 pilot pairs, excluded from the final comparison.
2. Estimate variance and throughput. Plan for 80% power at a 10 bb/100 effect,
   using `ceil((z(.975)+z(.8))^2 * s^2 / 0.1^2)` independent pairs.
3. Freeze the sample size in `plan.json` before collecting confirmation data:
   minimum 100,000, maximum 1,000,000 pairs. Pilot throughput can lower the
   maximum to fit 80% of the remaining runtime, but never below the minimum.
4. Calculate a two-sided Student 95% interval across pair averages. A lower
   bound above zero supports improvement; an upper bound below zero supports
   regression; otherwise the result is inconclusive.

No significance-based early stopping or re-use of pilot hands is allowed.
Power is an approximation based on estimated pilot variance. Capping the sample
may reduce power; both required and selected sample sizes and estimated
detectable effect are reported. An incomplete fixed sample has no confirmatory
claim even if its descriptive interval excludes zero. Zero sample variance is
explicitly treated as degenerate, not perfect certainty.

The primary receives 80% of the local evaluation time. The remainder is shared
between P4/P8 versus uniform-random and check/call controls (1,000 pilot pairs,
10,000 final pairs per match). Those comparisons are descriptive diagnostics,
not additional primary hypothesis tests. Identical seeds provide reproducible
deals for the controls; the report does not mistake them for independent seeds
of the training algorithm.

```bash
python -m hunl_escher.evaluate --a outputs/learning-4plus4/P8.pkl \
  --b outputs/learning-4plus4/P4.pkl --workers 8 \
  --output outputs/separate-head-to-head
```

## External benchmarks and failure isolation

Run the two provider jobs independently, preferably on the small evaluator VM.
Each job waits for verified policy publication. P4's pilot can start while
training continues. Once P8 is available it pilots P8, fixes the main sample size,
then alternates both policies in randomly ordered blocks of 100 hands.

Each provider uses 1,000 pilot hands per policy; the main sample targets 80% power
for a 10 bb/100 difference, with 10,000–20,000 **additional hands per policy**.
Independent-arm variance determines the sample size. There is no assumed remote
duplicate-deal facility. Raw winnings are measured in bb; GTO's primary score is
per-hand AIVAT chips divided by 100. Raw scores are reported alongside AIVAT.
Slumbot baseline-adjusted values remain in the journal but are not used as AIVAT.

Individual scores have 95% intervals. Differences use independent-arm Welch
97.5% intervals: Bonferroni adjustment for two external opponents. Overall
improvement claims are gated on the primary head-to-head passing; secondary
results alone remain supporting/exploratory evidence. We do not require beating
the external opponent: a less negative P8 score can constitute improvement.
GTO's reported engine version is checked before and after the main sample when
available. A known change invalidates the comparison; unknown versions are
recorded explicitly, including for Slumbot.

```bash
# Separate processes/terminals. Each may start before P4 exists.
python -m hunl_escher.benchmark --provider slumbot \
  --experiment outputs/learning-4plus4 --output outputs/learning-4plus4/benchmarks/slumbot
python -m hunl_escher.benchmark --provider gtow \
  --experiment outputs/learning-4plus4 --output outputs/learning-4plus4/benchmarks/gtow
```

The external budget is 24 wall hours **including waiting for policies**. Jobs
finish early if their fixed samples complete. The existing transport uses one
sequential session/provider, at least 500ms between requests, 30-second request
timeouts and bounded GET retry/backoff. Unknown POST outcomes are never blindly
retried. A failed provider stops its own job; it does not restart indefinitely or
affect training/the other provider. Missing/denied GTO access is
`SKIPPED_UNAVAILABLE`. Other provider failures are `INCOMPLETE_PROVIDER_ERROR`;
protocol/model errors are `EVALUATION_ERROR`. Durable journals retain completed
results and unresolved requests. Missing outcomes are never imputed as zero.

GTO requires `GTO_WIZARD_API_KEY`; it may affect the account leaderboard. Slumbot
uses anonymous play, with local checkpoint-specific scores. These runners do not
automatically restart interrupted API sessions, because a terminal response may
have been lost. Review the journal and use the existing explicit recovery tools.
An interrupted benchmark can be replaced by a new separately preregistered run;
do not pool selected partial outcomes as a completed fixed-sample test.

## Hardware preflight and Google Cloud

Start with an on-demand `c4-highmem-16` in `us-central1-a`: 16 vCPUs, 124 GiB,
200 GiB Hyperdisk Balanced, CPU-only. Dense replay allocation is about 33.5 GB;
full-process memory peaks are not yet certified. The separate external evaluator
is `c4-standard-4`, 15 GiB, with a 30 GiB Hyperdisk Balanced boot disk.

Default training is eight single-thread collection actors and four central
PyTorch threads, with collection and fitting in separate phases. Preflight tries
(workers, threads) = (8,4), (12,4), (8,2), (8,8), each in a separate process, within
30 minutes. It retains full network widths and batch sizes, but bounds replay
and traversals and uses 500 policy updates. This data is discarded. Selection
uses warm-iteration elapsed time and requires a 10% gain to replace the default.
An entirely unsuccessful preflight stops training. `--skip-preflight` keeps the
requested worker/thread configuration. Configuration is frozen across P4/P8.

This is an early-replay hardware proxy, not a hardware/configuration sweep for
playing strength or a full-capacity memory certification. Production stage
reports record process-tree RSS (which can double-count shared memory), host
memory availability, phase times and replay occupancy. Do not downsize to 60 GiB
based only on partially occupied replay or one short preflight measurement.

Prepare a launch bundle without making network calls or creating resources:

```bash
python tools/launch_learning_check.py \
  --project YOUR_PROJECT --bucket gs://YOUR_PRIVATE_BUCKET/hunl \
  --service-account YOUR_SERVICE_ACCOUNT \
  --run-name learning-4plus4-001 --output outputs/cloud-plan
```

The bundle contains a source-only archive, two startup scripts and `launch.json`.
It excludes `.env`, `.git`, outputs and credential files outside source packages.
Review it, then submit with the same arguments plus `--execute`, using a new
local output directory such as `outputs/cloud-submit`. Use a unique remote run
name. No existing bucket, IAM grant or account is created by this launcher.

Prerequisites: enabled Compute Engine/Storage APIs, C4 capacity/quota in the
chosen zone, an existing private bucket, permission to create VMs and act as the
specified service account. The VM service account needs object read/write/delete
access in the run bucket. Optional GTO access also needs Secret Manager access to
`gtowizard-benchmark-api-key` (override `--gto-secret`). An absent/inaccessible
secret does not block Slumbot. Keys never appear in startup scripts or archives.

The training VM has a 15-hour hard stop; the external VM has a 26-hour hard stop,
allowing boot/setup beyond their inner budgets. Startup scripts also shut down
when jobs exit. A manual VM restart does not silently launch the experiment again.
Stopped VMs retain disks for recovery; **delete them after verifying artifacts**
to end storage charges. Hard VM stops can interrupt uploads; disks are retained.
Cloud submission itself and live opponents have not been exercised in offline tests.

At planning rates $1.042712/hour and $0.19767/hour, the maximum configured VM
compute duration costs about $20.78; disk, network, provider charges and taxes
are additional. A $20–25 allowance is a planning estimate, not an enforced total
billing cap. Source: [Google Cloud pricing](https://cloud.google.com/products/compute/pricing/general-purpose?hl=en).
Run-duration flags: [Google Cloud documentation](https://docs.cloud.google.com/compute/docs/instances/limit-vm-runtime).

## Results

`experiment.json` records actual active/wall times, checkpoint hashes, diagnostics
and all local results. Each benchmark has a report, a frozen sample plan and a
durable hand journal. Generate/update the combined human-readable report after
downloading cloud artifacts or finishing external evaluation:

```bash
gcloud storage rsync --recursive gs://YOUR_PRIVATE_BUCKET/hunl/learning-4plus4-001 outputs/downloaded-run
python -m hunl_escher.experiment_report --experiment outputs/downloaded-run
```

Read `learning_report.md` and `learning_report.json`. Engineering diagnostics
include finite recorded metrics/exported weights, changed policy parameters,
critic versions, experience by player/street, replay occupancy, collection/fit
times, extraction/save costs and measured memory. These are distinct from the
statistical strength result. Confidence intervals quantify evaluation-hand
uncertainty for this training seed, not uncertainty across independent training runs.
