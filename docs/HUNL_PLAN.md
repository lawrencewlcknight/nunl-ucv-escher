# Work after baseline extraction

The current repository deliberately establishes a runnable, testable starting
point. It does not claim benchmark-ready play against Slumbot or GTO Wizard.

1. **Use the selected training bet menu.** Interface v2 now uses exact OpenSpiel
   `fullgame` states with seven default learner actions, explicit legal-size
   deduplication and off-menu exact bet ingestion. See [GAME_INTERFACE.md](GAME_INTERFACE.md).
   The menu remains a training abstraction. Keep it fixed for the production run
   and evaluate generalisation to off-menu histories.
   External public-state reconstruction and protocol adapters are implemented
   in [API_INTERFACE.md](API_INTERFACE.md). Verify the bundled engine's action-count
   ceiling against the benchmark rules before claiming identical game trees.
2. **Retain the validated feature interface.** The v5 four-street descriptors and
   their lightweight cost audit are implemented; see [STRATEGIC_FEATURES.md](STRATEGIC_FEATURES.md).
   No playing-strength gain is inferred from correctness tests or the
   microbenchmark alone; a feature sweep is not required before production.
3. **Profile on the intended training machine.** Establish memory, mature replay
   cost and iterations/nodes per wall-clock hour. Compare sequential and eight
   actor collection. Select checkpoint frequency and bounded retention. Verify
   installed wheel behaviour and exact continuation on the production runtime.
4. **Add evaluation.** Reusable local opponents, seat-balanced matches, independent
   evaluation RNG, per-hand payoff logs in bb, fixed evaluation sample budgets
   and confidence intervals. Random-opponent success is only an engineering check.
5. **Expand external evaluation after API smoke tests.** Slumbot and GTO Wizard
   adapters now share the exact policy input path, and include offline equivalence
   tests, sequential clients, durable logs and conservative timeout handling.
   Slumbot is smoke-tested live; GTO Wizard needs an approved key for its first
   live verification. Add statistically planned, seat-balanced evaluation runs
   and independent uncertainty estimates before comparing playing strength.
6. **Use the agreed production schedule.** See [BASELINE.md](BASELINE.md#production-schedule-and-monitoring).
   Replay-fill schedules restore 2,500 critic and 2,000 calibration updates at
   full occupancy. Regret fits remain 750 and policy extraction 20,000 updates.
   Monitor one production run; do not require costly configuration experiments.
7. **Add cloud orchestration after the local contract is stable.** New job names,
   clean run identities, source pinning, resource caps and restartable storage.
   Do not re-enable old FHP jobs by renaming their labels.

External reference points:

* [OpenSpiel universal poker implementation](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/games/universal_poker/universal_poker.cc)
* [Slumbot official API example](https://slumbot.com/sample_api.py)
* [GTO Wizard benchmark](https://benchmark.gtowizard.com/)
