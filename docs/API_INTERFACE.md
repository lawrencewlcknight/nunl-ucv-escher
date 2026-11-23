# Slumbot and GTO Wizard interface

Both clients feed the same `DecisionView` into the same encoder and saved
average-policy network used by OpenSpiel training. The feature contract is now
`hunl_strategic_public_state_v3`; old v2 policies require fresh training.
The original live API validation in `API_VALIDATION.json` records v2; the new
feature validation is recorded separately in `FEATURE_VALIDATION.json`. The runner loads no training critics, optimisers or replay buffers.
The seven-action menu (or the checkpoint's configured menu) still describes our
choices; exact opponent amounts are retained even when they are off-menu.

## Implementation

* `remote.py`: card/history parsing, seat and chip-unit translation, public-state
  reconstruction, action encoding and response continuity checks.
* `api.py`: synchronous HTTP clients, token/authentication handling, bounded
  read retries and conservative recovery from ambiguous mutations.
* `play.py`: seeded inference-only agents, bounded sequential sessions, a durable
  JSONL journal and a command-line entry point (`hunl-play`).
* `offline.py`: synthetic wire-format fixtures and equivalence checks against
  independently generated native OpenSpiel histories. No network calls.

`HUNLFeatureEncoder.encode_decision(view)` is shared by training and external
play. `LoadedHUNLPolicy.action_probabilities_from_view(view)` returns probabilities
for the legal menu indices. `RemoteHand.action(index)` converts that index into
an API action and, for raises, its cumulative contribution on the current street.

OpenSpiel uses player 0=button/small blind, player 1=big blind. Slumbot uses the
opposite position numbers. GTO Wizard seats are read from their position labels;
unknown labels fail explicitly. In both API histories, `bN` means the cumulative
amount on that street. OpenSpiel's native N means the cumulative contribution
over the whole hand. For example, after contributing 275 preflop, a flop bet
of 137 becomes native action 412. Check and call are distinguished by the amount
currently owed, despite sharing a network output index.

## Unknown opponent cards

Each API response is reconstructed independently from its public history. Two
arbitrary opponent cards disjoint from all currently visible cards are supplied
to OpenSpiel only so its native engine can replay the hand. They are not an
estimate of the opponent's hand, and never enter the policy features or logs.
Rebuilding from scratch permits a later board card to coincide with an earlier
temporary card without corrupting the state.

Only the hero's `DecisionView` leaves reconstruction. Tests vary these temporary
cards and check identical inputs and probabilities. Real showdown winnings come
from the remote server; the temporary cards' showdown outcome is never used.
Fold winnings can additionally be checked locally because they do not depend on
private cards. GTO Wizard's pot, stacks, legal actions and raise bounds are checked
against the reconstructed state. Any disagreement stops the run.

This retains the pinned OpenSpiel engine's limits, including its approximate
64-action-per-round ceiling and the 218-event feature bound. An incompatible
remote history is rejected, not silently truncated or rounded to a training bet.

## Lightweight commands

From the repository, using the installed Python environment:

```bash
python -m pip install -e '.[api,dev]'

# No API access: both protocol paths, both seats, all streets and off-menu bets.
python -m hunl_escher.play --opponent offline --hands 100 --seed 2026 \
  --output outputs/play/offline.jsonl

# Optional tiny trained checkpoint; only 32 trajectories, not a strong strategy.
python -m hunl_escher.train --smoke --iterations 1 --output outputs/play/tiny-policy

# Same test through the real checkpoint's neural inference.
python -m hunl_escher.play --opponent offline --hands 100 --seed 2026 \
  --agent checkpoint --checkpoint outputs/play/tiny-policy/policy_000001.pkl \
  --output outputs/play/offline-policy.jsonl

# Small anonymous live Slumbot session; default agent checks/calls.
python -m hunl_escher.play --opponent slumbot --hands 10 \
  --output outputs/play/slumbot.jsonl

# The same saved policy against Slumbot.
python -m hunl_escher.play --opponent slumbot --hands 10 --seed 2027 \
  --agent checkpoint --checkpoint outputs/play/tiny-policy/policy_000001.pkl \
  --output outputs/play/slumbot-policy.jsonl

# Once GTO_WIZARD_API_KEY is set securely to an approved researcher key:
python -m hunl_escher.play --opponent gtow --hands 10 --seed 2027 \
  --agent checkpoint --checkpoint outputs/play/tiny-policy/policy_000001.pkl \
  --output outputs/play/gtow-policy.jsonl
```

Always choose a new output filename: journals are never overwritten. Offline mode
uses a seeded mixture of check/call, menu actions and arbitrary native raises;
it checks both adapters at every decision. When a checkpoint is supplied it also
compares its probabilities exactly. It makes no claim about playing strength.

Live agent choices are `check_call`, `random` (uniform legal menu), `all_in`, and
`checkpoint` (sample the learned average policy, with the specified seed).

Slumbot defaults to anonymous play even if credential variables already exist.
Account play requires `--slumbot-login` plus `SLUMBOT_USERNAME` and
`SLUMBOT_PASSWORD`. GTO Wizard reads `GTO_WIZARD_API_KEY`; `--gto-key-env NAME`
selects another variable. Keys/passwords should not appear in command-line
arguments. GTO Wizard hands are genuine account evaluations and may affect its
leaderboard. Authenticated read-only requests and complete live-hand play through
this adapter have succeeded. The
[corrected P32 experiment](GTO_WIZARD_P32_EXPERIMENT.md) defines the benchmark
and requires a verified corrected refit before starting live hands.

### Local credentials with `.env`

Keep local credentials in the repository-root `.env`, which Git ignores.
For a new checkout, copy `.env.example` to `.env`, run `chmod 600 .env`, and
populate `GTO_WIZARD_API_KEY` locally. Do not commit the populated file.
The CLI does not automatically load `.env`; from the repository root, load it
into the current shell before running the play commands above:

```bash
set -a
source ./.env
set +a
```

Only source your own trusted file, since `source` executes shell code. Avoid
shell tracing (`set -x`) while loading credentials. After evaluation, remove
the key from the shell environment with `unset GTO_WIZARD_API_KEY`.
The matching Google Cloud Secret Manager secret is
`gtowizard-benchmark-api-key` in project `clever-overview-399515`.
Keep the local copy in sync when rotating the cloud secret.

## Network failures and recovery

Requests are sequential, with at least 0.5 seconds between starts and a default
180-second timeout for the GTO benchmark client, matching the
[official client](https://github.com/gtowizard-ai/researcher-api-client/blob/main/src/main.py).
The generic play CLI retains its configurable `--timeout` (30 seconds by default);
use `--timeout 180` for GTO Wizard. GET requests have at most three
attempts with bounded backoff. TLS verification stays enabled; the optional
`api` extra supplies a standard Certifi CA bundle for Python installations without
a usable default CA file. Credentials are never forwarded through redirects.

GTO POST actions and new-hand requests retry explicit HTTP 502, 503 and 504
responses, matching the official client's engine-busy policy: at most 32 attempts,
waiting 2, 4, 8, 16, then 32 seconds between attempts. The same request is repeated;
the policy does not choose a new action. The benchmark wall deadline still applies.
Other POST errors do not gain retries, and failed hands are not silently skipped.

For GTO HTTP 409 conflicts, errors include the method/path and a bounded JSON
`detail` string with credential values redacted. A 409 is not automatically
treated as engine busy or assumed to mean quota exhaustion. The benchmark
[FAQ](https://gtowizard.com/benchmark/faq) lists per-user/game limits of 10,000
hands/day, 25,000/week and 50,000/month; the daily reset is midnight UTC.

A lost response
might mean the server already applied the action. GTO Wizard can query
`GET /hands/in-progress`: the client continues only when the returned public
history proves the submitted action occurred in the same hand. If the state is
unchanged, inconsistent or missing, the client stops. A missing hand might have
finished, but that does not establish its payoff. Slumbot has no documented
read-only recovery endpoint, so an ambiguous POST stops its session immediately.
An errored client instance blocks further mutations.

The GTO adapter identifies the seat opposite `GTO Wizard` (or `gto_wizard`),
allowing the user's registered bot name. Terminal responses can reveal both
players' cards, including when the opponent folds before the user acts; these
hands must still be scored. Opponent cards are not used for policy decisions.

For a known unfinished GTO Wizard hand, an explicit new runner can resume it:

```bash
python -m hunl_escher.play --opponent gtow --resume-hand 123 --hands 1 \
  --agent checkpoint --checkpoint outputs/play/tiny-policy/policy_000001.pkl \
  --output outputs/play/gtow-resumed.jsonl
```

This reads the existing hand; it does not replay the previous action or claim
recovery of a missing terminal result. Review an unresolved action before starting
a separate resumed run. A newly seeded runner is not exact RNG continuation of
a previous live session. There is no automatic process-restart/resume mechanism.

The journal flushes and fsyncs action intent before sending it, then records the
validated observation/result. It includes seed, checkpoint hash, action menu,
public observations, raw chip/big-blind winnings and separately named adjusted
scores (`baseline_winnings` for Slumbot; `aivat_score_chips` for GTO Wizard).
Tokens, headers, passwords and opponent cards are excluded. Unresolved hands are
not counted as completed. Small-sample win rates are engineering diagnostics.

## Verification

```bash
python -m pytest -q
RUN_RAY_TESTS=1 python -m pytest -q -m ray
```

The API tests cover 500 seeded synthetic hands through both parsers, all four
streets and both seats, every legal outgoing menu action, off-menu opponent
bets, short all-ins and runouts, missing final separators, malformed responses,
visible-card collisions, temporary-card independence, learned-probability
identity, mock full sessions, rotating tokens, explicit hand resumption,
read-only timeout reconciliation, ambiguous-terminal outcomes, refusal to repeat
mutations, journal redaction and overwrite protection. Offline fixtures are
synthetic examples of the documented protocols, not captured live GTO responses.

The actual development smoke results are recorded in
[API_VALIDATION.json](API_VALIDATION.json). The raw local smoke logs and tiny
checkpoint are under `outputs/api-smoke/` and `outputs/api-smoke-policy/`; these
are ignored artifacts, not committed experiment results.

Official references:

* [Slumbot sample client](https://slumbot.com/sample_api.py).
* [Slumbot owner guidance on research usage](https://github.com/ericgjackson/slumbot2019/issues/49).
* [GTO Wizard researcher schema](https://researcher.gtowizard.com/openapi.json).
* [GTO Wizard official client](https://github.com/gtowizard-ai/researcher-api-client).
* [GTO Wizard benchmark and access application](https://benchmark.gtowizard.com/).
