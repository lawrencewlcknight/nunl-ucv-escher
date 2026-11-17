# P8 plus 24 active training hours

This is a continuation of the original 200bb, blinds 50/100, no-rake HUNL run.
Restore its **full P8 recovery**, including replay, optimisers, random states and
iteration counters. A playable policy alone cannot continue learning.

The coordinator executes the original archived trainer in an isolated working
directory with its original source fingerprint, Python/library runtime and
configuration. Strict resume checks remain enabled. Network architecture,
features, learning budgets, collection actors and Torch threads remain unchanged.
The existing production P8 contract uses eight actors and eight Torch threads.

## Training and recovery

| Additional active time | Approximate cumulative time | Policy |
|---|---|---|
| 6 hours | 14 hours | P14 |
| 12 hours | 20 hours | P20 |
| 18 hours | 26 hours | P26 |
| 24 hours | 32 hours | P32 |

Targets are relative to P8's actual recorded active time, at completed iteration
boundaries. Overshoot does not accumulate across stages. Exporting the average
policy, persisting state, uploading and evaluation do not consume active time.
Use the recorded timestamps and iteration counts when comparing throughput.

Each stage exports a playable average policy and full recovery. Preserve all
five policies (including P8), the original P8 run, and the newest continuation
recovery. Recovery shards upload before their latest pointer and policy marker;
the preceding continuation recovery is pruned only after successful publication.
Fresh subprocesses verify initial and final full-state restores, all shard
checksums, source/runtime identity and exact equality with playable policy weights.

`training_source.tar.gz` in the continuation output is the original trainer.
`source.tar.gz` is the new coordinator and evaluator. Future training must use
the former, the saved manifest and exact runtime. `resume.json` identifies these
artifacts. Keep full state shards and `training/latest.json`, not just P32.pkl.
Automatic restart of a failed cloud job is deliberately disabled; preserve its
disk and inspect its latest recovery before explicitly resuming.

## Local evaluation, after training

Primary: **P32 vs P8**, with a two-sided 95% Student interval over independent
duplicate-pair averages. Secondary: P14/P8, P20/P14, P26/P20 and P32/P26, each
with a 99% interval. The four 1% tests conservatively control their family error
at at most 4%; secondary improvement claims also require the primary to pass.
No assumption is made that successive policies improve monotonically.

Each matchup uses exchanged seats on the same predealt cards. A separate 10,000
pair pilot estimates variance/throughput; the confirmation sample is then frozen
before collecting any confirmation results. Target: 80% estimated power for a
10 bb/100 effect, 100,000–1,000,000 pairs per matchup. Time/sample caps, detectable
effect estimates and incomplete samples are reported. No significance-based
early stopping or pilot reuse. Confidence intervals cover evaluation-hand
uncertainty for this training seed, not exploitability or variation across seeds.

The local evaluator gets four wall hours: initially 40% for the primary, then
the remaining time shared across the adjacent comparisons. Child process limits
prevent a stalled inference pool keeping the training VM alive indefinitely.
`evaluation/report.json` and `evaluation/continuation_report.md` report results.

## Final-policy-only Slumbot

Only **P32** plays new Slumbot hands: 1,000 pilot hands followed by exactly
20,000 confirmation hands. The count is fixed before the pilot; the pilot
estimates precision but does not trigger changes to the sample size. Scores use
raw winnings in bb, with a descriptive 95% interval excluding pilot hands.
This cost-limited benchmark does not promise power to detect modest improvements.

The original P8 Slumbot report is an optional historical reference. It must
finish its original fixed sample, use raw Slumbot scores and match the P8
checkpoint identity. The P32-minus-P8 comparison uses an independent-arm Welch
interval, includes uncertainty in both estimates, and is always descriptive:
evaluation dates differ and the Slumbot version is unknown. An incomplete old
report is explicitly unavailable as a completed baseline. New P8 Slumbot hands
are not requested. GTO Wizard is not part of this experiment.

The API adapter retains existing pacing, request timeouts and durable journals.
Unknown POST outcomes stop the job and are never blindly retried or scored as
zero. Partial samples remain labelled incomplete. There is no automatic restart
or significance-based extension. Results go to `benchmarks/slumbot/report.json`,
`slumbot_report.md`, and `hands.jsonl`. The exact historical report is archived
alongside them when available.

## Cloud launch: two separate stages

Training and local evaluation use one `c4-highmem-16` VM, 200GB boot disk,
eight actors/eight Torch threads. Training has a 30-hour wall budget; local
evaluation has four hours. The VM has a 36-hour hard stop including setup.

Slumbot uses a separate `c4-standard-4` VM with a 30GB disk, a 24-hour benchmark
budget and a 26-hour hard stop. **It is launched by a second command after the
training run publishes readiness**, so no paid evaluator waits through training.
The readiness check also requires the final full-state reload verification and
completion of the local evaluation phase (which may report inconclusive or
incomplete matches). No new IAM permissions or background scheduler are needed.
Both VMs shut down on success or failure; stopped disks remain billable until
explicitly deleted after verifying artifacts.

Prepare the training bundle (no VM is created without `--execute`):

```bash
cd /Users/lawrenceknight/Documents/deep_cfr_v3/hunl-ucv-escher
HUNL_RUN="hunl-plus24-$(date -u +%Y%m%d-%H%M%S)"
python3 tools/launch_continuation.py \
  --project clever-overview-399515 --zone europe-west1-b \
  --bucket gs://clever-overview-399515-fhp-escher-results/hunl \
  --service-account fhp-escher-runner@clever-overview-399515.iam.gserviceaccount.com \
  --source-uri gs://clever-overview-399515-fhp-escher-results/hunl/hunl-4plus4-20261115-104829 \
  --expected-p8-sha256 b0712b9781b83331388ab4652c16a2ebd909885466c2890df70d040d32f9d410 \
  --run-name "$HUNL_RUN" --output "outputs/cloud-$HUNL_RUN"
```

For submission, add `--execute` to that invocation and use a new run name/output
directory. Keep the resulting `launch.json` path. This command submits only the
training VM. Source archives use an allowlist and exclude `.env`, credentials,
outputs and Git metadata. The source archive hash is checked during startup.
The trainer checks the exact saved Python and library versions before learning;
an incompatible base image fails safely rather than relaxing the contract.

After training/local evaluation finish, run this using the saved plan path:

```bash
python3 tools/launch_continuation_slumbot.py \
  --training-plan "outputs/cloud-$HUNL_RUN/launch.json" \
  --output "outputs/slumbot-$HUNL_RUN" --execute
```

The second launcher reads cloud readiness before creating anything. If called
too early, it exits. It pins the published P32 hash and uses the same source
archive as the training coordinator. A generation-guarded launch record rejects
duplicate submissions. If provisioning fails after reservation, inspect that
record and the VM before any deliberate retry; do not blindly rerun the job.

For a concise VM status from any terminal:

```bash
gcloud compute instances list --project=clever-overview-399515 \
  --filter='name~"hunl-plus24"' --format='table(name,status)'
```

## Local verification

### Recover a completed run after publication failure

`python -m hunl_escher.recover_continuation --workspace /var/lib/hunl-continuation
--uri gs://BUCKET/hunl/RUN` uses the existing VM workspace. It never trains.
Before starting it, replace the original startup script (which intentionally
shuts down on a second boot), protect the boot disk from automatic deletion,
and impose a six-hour VM stop limit. Keep the archived trainer unchanged.

The recovery verifies all five recorded policy identities and completed stage
targets, checks the final disk recovery, retries its upload, downloads a separate
cloud copy, and verifies every SHA256 plus a full eight-worker reload. Only after
both verifications succeed does it publish P32 and prune older continuation
recoveries. The original baseline run is a separate prefix and is preserved.
It then runs the existing four-hour local evaluation suite and publishes the
readiness marker for the separate Slumbot job. It keeps the original failure
record in `continuation-before-recovery.json`; `recovery-audit.json` records the
new status. A failed verification cannot publish readiness or prune recoveries.

Periodic publication stages diagnostic files only; recovery directories,
playable policies, and training pointers use the synchronous checkpoint
publisher. Cloud retention enumerates all older recovery directories within
the current run instead of relying on a single previous-state marker.

```bash
python -m pytest -q tests/test_continuation.py tests/test_cloud_recovery.py tests/test_storage_transfer.py
```

These offline tests create a small full-architecture recovery, continue four
stages using the archived trainer, verify its final reload, compare learner
state against uninterrupted updates, run all five local matches, and exercise
the Slumbot wire protocol against an OpenSpiel test server. They also cover
ambiguous requests, tampered policies, incomplete historical results and delayed
evaluator provisioning. Smoke results cannot make strength claims. Production
eight-actor restore and cloud/API throughput still need verification at launch.
