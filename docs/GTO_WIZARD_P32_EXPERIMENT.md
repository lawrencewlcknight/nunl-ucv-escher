# Corrected P32 against GTO Wizard

Evaluate the **objective-mass policy refit of P32**, using the first predeclared
fit seed, **1701**, against `HUNL 200BB`. This is an absolute external benchmark
of the corrected policy. It does not test whether the correction beats the
original policy; that comparison belongs to the separate sampler experiment.

The existing saved P32 is the **source**, not the corrected candidate:

- Run: `gs://clever-overview-399515-fhp-escher-results/hunl/hunl-plus24-20261116-074857`
- Recovery iteration: `1658`
- Original policy SHA-256: `73eec96257ba92f0c6b8e4fb43f040d8f88a7eb80c589dcffd71f567bb49f129`
- Candidate: `objective_mass_seed1701.pkl`, from a completed frozen-replay refit.

The refit changes group sampling and its mathematically required loss
compensation, preserving the expected weighted objective. It uses existing
replay, with no additional self-play. It is not yet established that this change
improves playing strength. Never substitute original P32 or a synthetic smoke
checkpoint for the candidate.

## Design

| Item | Setting |
|---|---|
| Game | Heads-up no-limit Hold'em, 200bb, blinds 50/100 |
| Candidate | P32 objective-mass refit, seed 1701 |
| Pilot | 1,000 fresh hands, excluded from final estimates |
| Confirmation | Exactly 20,000 new hands |
| Primary score | Per-hand `aivat_score / 100`, reported in bb/100 |
| Secondary score | Per-hand raw winnings, reported in bb/100 |
| Interval | Two-sided 95% Student interval over fresh confirmation hands |
| Action seeds | 671901 pilot, 671902 confirmation |
| Execution | One sequential client, one inference thread, existing 0.5s API pacing |
| Budget | 24 hours; report incomplete if exhausted |

The pilot estimates precision but cannot extend the fixed confirmation sample.
There is no significance stopping. Intervals describe evaluation uncertainty for
one fixed checkpoint and opponent, not training-seed uncertainty, exploitability,
or proof of improvement over P32. Even a complete sample may be imprecise.

The runner checks the account's reported GTO Wizard engine version before play,
after the pilot, after each 100 hands, and at phase completion. A detected change
stops and invalidates the run. A null pre-pilot version on an unused account is
allowed; version verification requires non-null, identical checks from the end
of the pilot through confirmation. These are account-level observations, not
per-hand engine attestations. Use the account exclusively during this run.
The account's cumulative `/results` scores and leaderboard are never used as
the experiment sample, because they can include other policies and pilot hands.

Existing unfinished hands are recorded as excluded and left untouched. Only
fresh hands started by this runner count. In particular, hand `8906022` observed
during API validation has unknown policy provenance and must not be included.

## Prepare the corrected policy

See [the frozen-replay sampler experiment](POLICY_SAMPLER_EXPERIMENT.md) for
the bounded cloud launcher. Its `fit/` output is accepted by this benchmark.
The GTO benchmark is independently requested, so it does not require the
positive local-result gate used for the separate Slumbot follow-up.

For local execution with adequate memory and disk, use the existing refit tools:

```bash
cd /Users/lawrenceknight/Documents/deep_cfr_v3/hunl-ucv-escher

python3 -m hunl_escher.policy_sampling_experiment prepare \
  --source-state gs://clever-overview-399515-fhp-escher-results/hunl/hunl-plus24-20261116-074857/training/state_001658 \
  --output outputs/policy-sampler-source-p32

# The matching P32.pkl is already present locally; its hash is pinned above.
python3 -m hunl_escher.policy_sampling_experiment fit \
  --state outputs/policy-sampler-source-p32 \
  --policy outputs/policy-sampler-source-p32/P32.pkl \
  --output outputs/policy-sampler-p32 --seeds 1701 1702 --threads 8
```

Preparing the replay downloads approximately 18.7 GB plus metadata. The refit
has material RAM/CPU requirements; the existing cloud design uses a
`c4-standard-16` with a 100 GB disk and six-hour hard stop. This benchmark itself
only needs the small policy files and their `experiment.json`. Preserve all
matched refit files, since verification checks the complete refit manifest.

## Verify and run

**Current execution limitation (19 November):** the launch examples below start
a fresh 21,000-hand experiment and do not resume an earlier run. GTO Wizard's
[published limits](https://gtowizard.com/benchmark/faq) are 10,000 hands/day,
25,000/week and 50,000/month per user and game, resetting at midnight UTC
(weekly on Monday, monthly on the first). Do not restart the full experiment
unchanged: run `hunl-gtow-p32-20261118-143528` already saved its 1,000 pilot and
8,922 confirmation hands. Finishing the original sample requires 11,078 further
confirmation hands over at least two daily quota windows, with a continuation
that preserves the existing sample and policy/version provenance. That
continuation is not implemented by the current launcher. A larger VM or generic
HTTP 409 retries would not solve this scheduling limitation.

Load the existing trusted, Git-ignored local credentials without printing them:

```bash
set -a
source ./.env
set +a

# Read-only: verify candidate identity, load inference model, authenticate,
# validate existing public hand state, and read account version information.
python3 -m hunl_escher.gtow_p32 \
  --experiment outputs/policy-sampler-p32 \
  --output outputs/gtow-p32-preflight --preflight

# Live: creates and plays fresh hands on the researcher account.
python3 -m hunl_escher.gtow_p32 \
  --experiment outputs/policy-sampler-p32 \
  --output outputs/gtow-p32-benchmark --max-seconds 86400

unset GTO_WIZARD_API_KEY
```

Use new output directories for each invocation. The runner rejects incomplete
refits, uncorrected policies, synthetic production inputs, the wrong P32 source,
changed file hashes, or a different selected fit seed before any API calls.

## Outputs and failure handling

- `design.json`: fixed sample, source/refit/checkpoint hashes, code hashes and seeds.
- `hands.jsonl`: durable action intent, observations and completed-hand results;
  excludes credentials and opponent private cards.
- `pilot_precision.json`: pilot-only precision estimate for the fixed sample.
- `report.json`: progress, excluded hand IDs, version checks, separate pilot/main
  AIVAT and raw scores, completed counts, estimates and failure status.

The existing adapter uses a named research User-Agent. It has authenticated
successfully against the live API; default Python-urllib requests encountered
Cloudflare rejection in earlier troubleshooting. TLS verification stays enabled.

Unknown POST outcomes are never blindly retried or scored as zero. The existing
client may resolve an action through an authoritative in-progress read only when
public history proves that action was applied. Otherwise it stops. There is no
automatic restart or journal overwrite. The wall budget is checked before each
request; an already-running bounded request may finish after the deadline.

## Google Cloud Batch launch

The dedicated launcher submits one Google Cloud Batch job, using the same
managed-job model as the Leduc and FHP experiments. It requests one standard
`c4-standard-2` worker (2 vCPUs, 7 GB RAM), a 30 GB Hyperdisk Balanced boot disk,
the Batch Debian 12 image, one inference thread and no GPU. The single task
reserves 2 vCPUs and 6 GiB of memory. The benchmark budget is 24 hours; the task
limit is 26 hours including setup and publication, excluding queue time.
The main runnable has a 25.5-hour limit, leaving time for the final upload
runnable. Task retries are disabled (`maxRetryCount: 0`) because replaying a live
API action after an uncertain failure could duplicate play. This is a standard
provisioning job, not a Spot job.

The default inputs are pinned to the completed refit
`hunl-sampler-p32-20261117-170440`, manifest SHA-256
`8df5ecc1e088f38e6dd7f1918e71b30d1869d76b17144f61b6642101281590b5`
and candidate SHA-256
`add9bf970e9979be6b74c0548b084d1af19e1c8b334bd4ef326f192ceafabe29`.
The launcher packages the current allowlisted source, including the GTO worker;
it does not reuse the older refit source archive, which predates this runner.

For a fresh experiment, the existing submission syntax is below. This is not
a resume command; see the usage-limit restriction above before submitting.

```bash
cd /Users/lawrenceknight/Documents/deep_cfr_v3/hunl-ucv-escher
GTO_RUN="hunl-gtow-p32-$(date -u +%Y%m%d-%H%M%S)"

python3 tools/launch_gtow_p32.py \
  --run-name "$GTO_RUN" \
  --output "outputs/cloud-$GTO_RUN" \
  --project clever-overview-399515 \
  --region europe-west1 \
  --gto-secret gtowizard-benchmark-api-key \
  --execute
```

Omit `--execute` to prepare a local bundle only. Each invocation needs a new
output directory. The bundle contains `job.json`, the two runnable scripts,
`launch.json` and the source archive. The defaults use the existing results bucket under `hunl/`
and service account
`fhp-escher-runner@clever-overview-399515.iam.gserviceaccount.com`.
The service account needs the usual Batch Agent Reporter and Logs Writer roles,
access to the input/output bucket and Secret Manager Secret Accessor on
`gtowizard-benchmark-api-key`. The submitting identity needs permission to create
Batch jobs and act as this service account, as in the existing Batch workflow.
The launcher does not change
IAM permissions. The key is retrieved into memory on the VM; it is never included
in launch metadata, source archives, command arguments or result files.

Before submission, the launcher checks the remote refit manifest hash. On the
VM, the worker downloads and verifies the four small refit policies, then runs
one benchmark loop. Its initial authenticated GET checks must pass before play.
The worker no longer invokes a separate duplicate preflight. There is
no training, no replay download and no Slumbot job. Generation-guarded uploads
prevent reuse of a run prefix. If submission fails, inspect its launch record
and Batch job before deliberately trying again; do not blindly create another run.

Reports and complete-line hand-journal snapshots are uploaded periodically, with
a two-minute wait between uploads. The publisher copies local files to a stable
snapshot before transferring them. Final publication is mandatory on worker exit;
SIGTERM requests an orderly journal close and upload. A separate `alwaysRun`
runnable also attempts to upload results and the setup log after the main runnable
fails. Storage copies may retry within bounded deadlines. GTO requests follow
the provider's engine-busy retry policy described below.

Batch cleans up the worker VM and disk. Abrupt VM loss or the hard task deadline
can prevent final publication, so the journal tail since the last successful
snapshot can be lost; there is no automatic resume. The fallback upload after a
forced termination may contain an incomplete final journal line. Check Batch
status, `cloud_status.json` and the benchmark report before treating a run as
complete. See Google's [Batch job lifecycle](https://docs.cloud.google.com/batch/docs/create-run-job)
and [runnable and timeout reference](https://docs.cloud.google.com/batch/docs/reference/rest/v1/projects.locations.jobs).

```bash
gcloud batch jobs describe "$GTO_RUN" \
  --project clever-overview-399515 --location europe-west1 \
  --format='value(status.state)'

gcloud storage cat \
  "gs://clever-overview-399515-fhp-escher-results/hunl/$GTO_RUN/cloud_status.json"

gcloud storage cp \
  "gs://clever-overview-399515-fhp-escher-results/hunl/$GTO_RUN/benchmark/report.json" \
  "outputs/cloud-$GTO_RUN/report.json"
```

The reports appear after worker setup and input verification. Setup and worker
output are available in Cloud Logging through the Batch job's logs; setup output
is also copied to `startup.log` by the final runnable.

### Bootstrap failure corrected on 18 November 2026

Job `hunl-gtow-p32-20261118-095722` failed after 38.5 seconds of task execution
while creating its Python virtual environment. The source archive checksum passed;
`python3 -m venv` then reported that it could not determine the running interpreter
path. The benchmark worker was never reached, so this job made no GTO API calls
and played no hands. The final runnable successfully saved `startup.log`.

This matches the earlier FHP Batch bootstrap failure. The failed GTO log did not
record its inherited PATH, so the exact environment cannot be reconstructed.
Both runnables now export a standard Linux PATH and preserve additional entries.
Bootstrap uses `/usr/bin/python3 -I` explicitly, and pip and worker invocations
also use isolated Python mode. Setup logs the interpreter identity and named
stages; failures report their stage, line and exit code without dumping the
environment. See [Python's interpreter-path check](https://github.com/python/cpython/blob/3.11/Lib/venv/__init__.py).

Local regression checks exercise real venv creation with absent/empty PATH and
invalid inherited Python environment variables, child-process PATH export, and
fail-fast stage reporting. They do not establish that a replacement job has
completed in GCP. To retry, rerun the launch command above with a fresh run name
and output directory so it generates the corrected Batch configuration; the old
prepared `job.json` files still contain the failed bootstrap.

### Terminal-hand parser correction — second 18 November attempt

Job `hunl-gtow-p32-20261118-101817` passed setup, checksum verification and
authenticated preflight. It recorded one pilot hand, then stopped with
`Cannot identify our seat without prior hand context` on the next new hand.
The account reported two completed hands and only the pre-existing unfinished
hand. The failed second response was not retained, so its exact contents cannot
be recovered through the documented API.

Live diagnostics confirmed that terminal responses expose both players' cards
and use player names `ucv-escher` and `GTO Wizard`. The old parser assumed that
only the user's cards were visible, with a fallback requiring the literal name
`user`. The corrected parser identifies the seat opposite the named GTO Wizard
player. Regression checks cover both seats, reversed player order, and an
opponent folding immediately on hand creation; such hands count toward the
sample without a user action. Unknown ambiguous identities still fail closed.

The Batch setup was compared with
`fhp_ucv_escher/fhp-ucv-escher-experiments/gcp/submit_batch_experiment.sh`.
Setup already succeeded in this attempt. Following its explicit experiment
diagnostics pattern, the GTO worker now prints its result and failure reason into
Cloud Logging and includes them in `cloud_status.json`. The adapter's locally
defined protocol error is also retained in `benchmark/report.json`.

The benchmark client's request timeout is now 180 seconds, matching the
[official GTO client](https://github.com/gtowizard-ai/researcher-api-client/blob/main/src/main.py).
An earlier diagnostic encountered HTTP 503. At this stage the client stopped
on all server-side POST errors; the fourth-attempt fix below corrects that
handling for the provider's documented engine-busy responses.

After the fix, a separate live check completed ten fresh hands using the pinned
corrected P32 policy with no recoveries. Its journal is
`outputs/gtow-protocol-fixed-20261118/hands.jsonl`. These are interface-validation
hands, excluded from the production experiment. The full Batch experiment has
not been resubmitted. Use a fresh run name to package and submit the updated
adapter; do not reuse a previously generated source archive or `job.json`.

### Package-manager contention — third 18 November attempt

Job `hunl-gtow-p32-20261118-105114` failed in `system_packages` after five seconds,
before downloading the source, loading the policy or contacting GTO Wizard.
The saved setup log identifies `unattended-upgr` as the holder of
`/var/lib/dpkg/lock-frontend`; apt exited with code 100.

The generated install command now sets `DPkg::Lock::Timeout=600`, so apt waits
up to ten minutes for the updater to release the lock. It uses the package
manager's own locking mechanism, without removing lock files or terminating
system updates. `--no-upgrade` preserves already-installed packages such as the
Cloud SDK, and `Acquire::Retries=3` permits bounded package download retries.
These settings apply only to package setup. Debian's [apt lock implementation](https://github.com/Debian/apt/blob/main/apt-pkg/deb/debsystem.cc)
documents the lock-timeout setting in use.

The separate diagnostic `hunl-gtow-bootstrap-check-20261118-v1` uses the same
generated setup on Batch Debian 12 and `c4-standard-2`. It deliberately holds
the dpkg frontend lock for twelve seconds, requires the install command to wait,
then installs the dependencies and verifies all pinned refit artifacts. It
does not retrieve the API key or make GTO API calls. Its local bundle is
`outputs/gtow-bootstrap-check-20261118/`; its setup log is published under the
matching run name in the usual results bucket.
The diagnostic finished with Batch status `SUCCEEDED`. All 24 cloud-launcher
tests also passed. This validates setup and policy loading in GCP; it does not
claim that the full 21,000-hand scoring experiment has completed or that the
external provider cannot interrupt a later run.

### Engine busy — fourth 18 November attempt

Job `hunl-gtow-p32-20261118-124444` passed setup, loaded the corrected policy,
passed authentication and completed five pilot hands. It then stopped on
`POST returned HTTP 503; mutation was not retried`. No confirmation hands ran.

The official [GTO client](https://github.com/gtowizard-ai/researcher-api-client/blob/main/src/main.py)
and [retry predicate](https://github.com/gtowizard-ai/researcher-api-client/blob/main/src/utils.py)
treat HTTP 502, 503 and 504 as engine-busy responses. The benchmark now follows
that pattern: repeat the same request up to 32 total attempts, waiting
2, 4, 8, 16, then 32 seconds between attempts. The 24-hour benchmark deadline
also bounds retries. Timeouts and lost responses still use conservative recovery;
authentication errors are not retried. Exhausted retries stop the run as incomplete
instead of silently dropping hands from the sample.

The cloud worker now calls the benchmark once. The separate preflight invocation
was redundant because the benchmark already checks authentication and policy
provenance before its first hand. The successful FHP-style Batch setup and result
publication remain in place.

The 91 targeted tests pass, including identical-request retries for both hand
creation and actions, the 32-attempt limit, the backoff cap, wall deadlines,
non-retryable authentication errors, and authentication checks before normal play.
A separate 50-hand Batch diagnostic, `hunl-gtow-busy-check-20261118-v1`, uses the
same generated setup and corrected policy. Its local bundle is
`outputs/gtow-busy-check-20261118/`; validation hands are excluded from the
production sample.

The diagnostic finished with Batch status `SUCCEEDED` after 164.6 seconds of
task runtime. It completed 50 unique hands with AIVAT scores and recovered from
five real HTTP 503 responses using the new retries. The downloaded evidence is
`outputs/gtow-busy-check-20261118/live-report.json` and `live-hands.jsonl`.
Both pre-existing unfinished hands were excluded. This validates the live API
failure path and GCP setup; the full 21,000-hand experiment has not been
resubmitted. Use the launch command above with a new run name to package the fix.

### Daily usage limit — fifth 18 November attempt

Job `hunl-gtow-p32-20261118-143528` ran for 21,603.8 seconds (about six hours).
It completed 1,000 pilot and 8,922 confirmation hands, then received HTTP 409
on the next `POST /hands`. The last hand, `9083576`, was terminal and journaled;
no experiment hand was left unresolved at the failure boundary. The saved
report and all 9,922 unique completed-hand records match exactly, including
every per-hand AIVAT score. GTO engine version checks remained at version 2.

The account began with 77 completed hands and ended with 9,999. A read-only
check on 19 November found only the same two excluded unfinished hands. A
single bounded diagnostic hand then completed successfully after the UTC daily
reset; that hand is excluded from the experiment. Combined with the published
10,000/day limit, this strongly supports daily quota exhaustion. The original
409 response body was not retained, so its precise server explanation cannot
be recovered or asserted as directly observed.

GTO HTTP 409 handling now retains a bounded JSON `detail` string, redacts
credential values, and records the request method/path. It does not retry a
conflict or label every 409 as a quota error; hand-state conflicts are also
possible. Non-JSON responses and non-string details remain suppressed.

Evidence is preserved locally under `outputs/gtow-failure-20261118-143528/`
(`report.json`, `hands.jsonl`, `startup.log`, `account-check.json`, and the
separate `single-probe.jsonl`) and in the original Cloud Storage run prefix.
The completed scoring sample is descriptive and incomplete, not a finished
20,000-hand experiment. No new full Batch job was submitted during diagnosis.

## Verification and current readiness

```bash
python3 -m pytest tests/test_gtow_p32.py tests/test_gtow_p32_cloud.py tests/test_remote.py -q
```

Tests exercise the actual GTO wire adapter against OpenSpiel-generated server
responses: corrected-policy provenance, fresh-hand play, excluded existing hands,
pilot exclusion, AIVAT conversion, read-only preflight, opponent version changes,
ambiguous requests, deadlines and overwrite protection. Synthetic tests are
explicitly labelled and cannot make live API calls through the CLI.

The corrected refit is now complete. All four downloaded policies passed their
manifest checksum checks, and `objective_mass_seed1701.pkl` passed the runner's
production P32 provenance checks. Offline cloud tests cover the Batch resource
configuration and time limits, secret handling, preflight failures, artifact
hashes, periodic journal snapshots, interruption and mandatory publication.
The fifth Batch attempt saved 1,000 pilot and 8,922 confirmation hands. The
original fixed sample remains incomplete. Separate diagnostic hands are not
production benchmark observations.
