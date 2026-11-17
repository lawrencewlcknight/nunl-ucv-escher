# No-Limit frozen-replay policy sampler experiment

Test whether changing grouped-policy sampling extracts a stronger playable policy
from learning already completed. **No additional self-play is collected.** The
production sampler, networks and ongoing continuation remain unchanged.

## Design

Use the same replay, action menu, encoder, network, Adam learning rate, batch
size, update budget, validation split and checkpoint selection in both arms:

* `uniform`: existing uniform group sampling without replacement per batch,
  with the existing per-group loss weights.
* `objective_mass`: independent draws with replacement proportional to summed
  replay weights. Multiply each sampled loss by the mean weight of the training
  groups, preserving the expected objective and its scale. Do not weight twice.

The shared sampler implementation is identical to Limit Hold'em. Unlike Limit,
the reader preserves the checkpoint's variable action menu (seven actions in
production), rather than assuming fold/call/raise only.

Fit seeds 1701 and 1702 give four policies. Each matched pair starts from exactly
the same fresh weights; source reset-each-fit semantics are preserved. Record
training and validation CE, target entropy, KL, action-probability error by
street, fitting time and preflop sample/weight variability. Validation selects
the fitting duration, so its loss is not an independent confirmatory endpoint.
No architecture or learning-rate changes, full-data refit, or best-seed selection
are introduced into this experiment.

The saved No-Limit P32 diagnostics already show preflop accounts for about
66.8% of replay weight but only 8.27% of unique groups: roughly 169 preflop draws
per 2,048-example uniform batch versus 1,369 expected under mass sampling, before
the validation partition. The experiment measures the actual training partition;
these figures are not a claim of a demonstrated strength improvement.

## Saved P32 source

Run: `gs://clever-overview-399515-fhp-escher-results/hunl/hunl-plus24-20261116-074857`.
Recovery: `training/state_001658`; matching policy: `training/policy_001658.pkl`
(also exported as `P32.pkl`). Policy SHA-256:
`73eec96257ba92f0c6b8e4fb43f040d8f88a7eb80c589dcffd71f567bb49f129`.

Only metadata and four policy arrays are required: eight million rows with 569
features, seven-action targets, masks and iteration weights. About **18.7 GB**
decimal, plus 25 MB metadata. The P32 policy hash, game, encoder, configuration
and all saved policy weights have been checked against its checkpoint metadata.
The loader verifies every selected array's checksum and shape before read-only
memory mapping. It never loads critic replay or starts the original solver.
This partial download cannot be used as a full training-resume checkpoint.

## Stage 1: local paired-deal test

For each independent deal, run both seats for each matched fit seed, then average
the scores across seats and fit seeds. One statistical observation is the entire
shared-deal group, not each hand or each fit seed. The primary comparison is
corrected minus uniform with a two-sided 95% interval. Per-fit-seed results are
descriptive. Inference is conditional on this one replay source and these fixed
refits, not across independent training runs or an exploitability estimate.

An excluded 10,000-group pilot estimates variance and throughput. Before main
play, fix a sample of 100,000–1,000,000 groups, targeting approximately 80% power
for 10 bb/100, consistent with existing No-Limit local evaluations. A two-hour
budget caps the sample. Report estimated detectable effect, sample caps and
incomplete runs; do not stop on significance or treat an underpowered null as
equivalence. Publish Slumbot eligibility only if the complete non-smoke primary
interval is strictly positive.

Launch on a separate c4-standard-16, eight fitting threads and eight evaluation
workers. The 100 GB disk accommodates the larger replay. No GPU is needed. The
VM uploads results, shuts down on exit and has a six-hour safety cap, not an
expected runtime. Stopped boot disks remain billable until explicitly deleted.

```bash
cd /Users/lawrenceknight/Documents/deep_cfr_v3/hunl-ucv-escher
RUN="hunl-sampler-p32-$(date -u +%Y%m%d-%H%M%S)"
python3 tools/launch_policy_sampler.py \
  --project clever-overview-399515 --zone europe-west1-b \
  --bucket gs://clever-overview-399515-fhp-escher-results/hunl \
  --service-account fhp-escher-runner@clever-overview-399515.iam.gserviceaccount.com \
  --run-name "$RUN" \
  --source-uri gs://clever-overview-399515-fhp-escher-results/hunl/hunl-plus24-20261116-074857 \
  --expected-policy-sha256 73eec96257ba92f0c6b8e4fb43f040d8f88a7eb80c589dcffd71f567bb49f129 \
  --iteration 1658 --output "outputs/cloud-$RUN" --execute
```

Omit `--execute` for a local launch bundle without a VM. The launcher packages
the current allowlisted source, excluding outputs, credentials and `.env` files.
Review outstanding source edits before packaging. Use a fresh name/output directory.
Results are under the run's `fit/` and `evaluation/` cloud prefixes. The four
playable policies retain source training provenance plus refit identities.
`cloud_status.json` distinguishes completion from errors; `slumbot_ready.json`
distinguishes `READY` from `NOT_ELIGIBLE`. A stopped VM alone does not prove success.

## Stage 2: separately invoked Slumbot follow-up

Use the **first predeclared fit seed, 1701**, for both arms. Never choose the
best-performing fit seed. The follow-up verifies the local result and hashes
before making any live API calls. It is not started automatically by stage 1.

Collect 1,000 pilot hands and then a fixed 20,000 new confirmation hands **per
arm**, interleaving randomly ordered 100-hand blocks. Both policies receive fresh
independent server deals, not duplicate deals. Pilot hands are excluded; the
pilot estimates detectable effect but does not change sample size. Use raw
winnings and an independent-arm Welch 95% interval for corrected minus uniform.
Do not compare only the candidate with an old historical Slumbot result.

The live API does not expose a verified Slumbot version, so comparisons are
explicitly descriptive when version stability cannot be verified. The local
primary remains the confirmatory result. The fixed external sample is cost
limited and does not promise power for small gains. A 24-hour wall budget can
leave it incomplete, especially because both arms need fresh hands. Preserve
partial results and never retry ambiguous POST outcomes or score them as zero.

Once the local result is eligible, explicitly launch the smaller c4-standard-4
with a 30 GB disk, 24-hour benchmark budget and 26-hour VM safety cap:

```bash
python3 tools/launch_policy_sampler_slumbot.py \
  --local-plan "outputs/cloud-$RUN/launch.json" \
  --output "outputs/slumbot-$RUN" --execute
```

This uses the original stage-1 source archive and downloads only the small
policy/result files, not replay. It requires no GTO Wizard key. API pacing,
timeouts, sanitized journals and off-menu action translation use the existing
Slumbot adapter. Results go under `slumbot/benchmark/` in the same run prefix.

## Manual execution and verification

With this repo's Python environment installed, the equivalent stages are:

```bash
python -m hunl_escher.policy_sampling_experiment prepare \
  --source-state gs://clever-overview-399515-fhp-escher-results/hunl/hunl-plus24-20261116-074857/training/state_001658 \
  --output outputs/policy-sampler-source-p32

gcloud storage cp \
  gs://clever-overview-399515-fhp-escher-results/hunl/hunl-plus24-20261116-074857/P32.pkl \
  outputs/policy-sampler-source-p32/P32.pkl

python -m hunl_escher.policy_sampling_experiment fit \
  --state outputs/policy-sampler-source-p32 --policy outputs/policy-sampler-source-p32/P32.pkl \
  --output outputs/policy-sampler-p32 --seeds 1701 1702 --threads 8

python -m hunl_escher.policy_sampling_evaluate \
  --experiment outputs/policy-sampler-p32 --output outputs/policy-sampler-p32-evaluation

python -m hunl_escher.policy_sampling_slumbot \
  --experiment outputs/policy-sampler-p32 --evaluation outputs/policy-sampler-p32-evaluation \
  --output outputs/policy-sampler-p32-slumbot
```

Use `prepare --metadata-only` to inspect required files before the large transfer.
Only load trusted checkpoint metadata (PyTorch/pickle). Existing verified arrays
are reused. Outputs must be new directories. No production run was launched as
part of implementing this experiment.

```bash
python -m pytest tests/test_policy_sampling_experiment.py tests/test_policy_sampling_cloud.py -q
```

Tests cover sampler mathematics, unchanged uniform fitting/optimizer/RNG behavior,
seven-action replay, checksums, matched initialization, playable exports, local
duplicate-deal aggregation, pilot exclusion, gate enforcement, two fresh offline
Slumbot arms, ambiguous outcomes, cloud publication and bounded launch scripts.
Smoke tests use an OpenSpiel-backed mock server and cannot make live API calls.
