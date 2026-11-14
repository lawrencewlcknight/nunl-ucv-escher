"""Small reproducible feature-cost audit; no training, network or strength claim."""
import argparse
import json
import platform
from time import perf_counter

import numpy as np

from .features import HUNLFeatureEncoder, LegacyHUNLFeatureEncoder, _canonicalise_suits
from .game import load_hunl_game
from .strategic_features import _card_features, _board_table


def audit(hands=100, seed=20261114):
    if hands < 1:
        raise ValueError("hands must be positive")
    game, encoder = load_hunl_game(), HUNLFeatureEncoder()
    legacy = LegacyHUNLFeatureEncoder()
    rng = np.random.default_rng(seed)
    states = []
    for _ in range(hands):
        state = game.new_initial_state()
        while not state.is_terminal():
            if state.is_chance_node():
                state.apply_action(int(rng.choice(state.legal_actions())))
            else:
                states.append(state.clone())
                state.apply_action(1)  # all four streets, both seats

    def prefix_only(state):
        view = state.decision_view()
        policy = np.concatenate((*_canonicalise_suits(encoder._cards(view.hole_cards),
            *encoder._board(view)), encoder._context(view))).astype(np.float32)
        critic = np.concatenate((*_canonicalise_suits(
            *[encoder._cards(cards) for cards in state.critic_hole_cards()],
            *encoder._board(view)), encoder._context(view))).astype(np.float32)
        return policy, critic

    def enhanced(state):
        return encoder.information_state(state), encoder.full_state(state)

    # Warm only native-state caches equally before timing either representation.
    for state in states:
        state.decision_view()
    elapsed = {}
    _card_features.cache_clear()
    _board_table.cache_clear()
    for name, fn in (("v2_prefix_only", prefix_only),
                     ("v3_first_stream", lambda s: (legacy.information_state(s), legacy.full_state(s))),
                     ("v5_first_stream", enhanced), ("v5_repeated_stream", enhanced)):
        start = perf_counter()
        for state in states:
            fn(state)
        elapsed[name] = perf_counter() - start
    # A hand normally revisits the same cards at multiple betting/learner nodes.
    # This checks that the old exact information is retained, outside timing.
    for state in states:
        old, new = prefix_only(state), enhanced(state)
        for before, after, layout, size in zip(old, new,
                (encoder.policy_layout, encoder.full_state_layout),
                (encoder.policy_prefix_size, encoder.critic_prefix_size)):
            split = layout.card_size + layout.strategic_size
            np.testing.assert_array_equal(before, np.concatenate((after[:layout.card_size],
                after[split:split + size - layout.card_size])))
    a, p, c = encoder.action_size, encoder.policy_size, encoder.full_state_size

    def replay_bytes(policy, critic):
        reservoir = 3 * 1_000_000 * 4 * (policy + 2 * a + 1)
        circular = 1_000_000 * (4 * (2 * critic + policy) + a + 8)
        return reservoir + circular

    return dict(kind="feature_engineering_microbenchmark_not_training_throughput",
        seed=seed, hands=hands, decisions=len(states), machine=platform.machine(),
        python=platform.python_version(), seconds=elapsed,
        microseconds_per_policy_plus_critic={name: value / len(states) * 1e6 for name, value in elapsed.items()},
        encoder=encoder.metadata(),
        primary_replay_bytes_v2=replay_bytes(encoder.policy_prefix_size, encoder.critic_prefix_size),
        primary_replay_bytes_v3=replay_bytes(legacy.policy_size, legacy.full_state_size),
        primary_replay_bytes_v5=replay_bytes(p, c),
        calibration_feature_growth_bytes=1_000_000 * 4 * (p - encoder.policy_prefix_size),
        memory_scope="primary replay excludes calibration, model/fit tensors, actors, caches and checkpoint copies; not RSS",
        cache_info={"cards": _card_features.cache_info()._asdict(), "boards": _board_table.cache_info()._asdict()},
        timing_scope="paired policy+critic construction only; one check/call stream, no neural inference or learner fitting")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hands", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261114)
    args = parser.parse_args()
    print(json.dumps(audit(args.hands, args.seed), indent=2))


if __name__ == "__main__":
    main()
