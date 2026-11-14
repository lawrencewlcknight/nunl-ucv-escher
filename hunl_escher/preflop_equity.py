"""169-class preflop cache using the limit encoder's fixed sampling contract.

This is uniform-opponent showdown equity, never a betting-conditioned range.
No global RNG is read or advanced; entries survive general card-cache eviction.
"""
from functools import lru_cache
from hashlib import blake2b, sha256
import json
from pathlib import Path

import numpy as np

SAMPLES = 4096
SAMPLING_ID = "canonical_visible_cards_blake2b_pcg64_partial_fisher_yates_v1"
TABLE_ID = "holdem_uniform_preflop_pcg64_4096_v1"
TABLE_SHA256 = "460df22fe919ba899d98229b17a53ea0cc12694feed6ed2c6f83dfbaaef94e2b"


@lru_cache(maxsize=1)
def _table():
    data = (Path(__file__).parent / "data/preflop_equity_v1.json").read_bytes()
    if sha256(data).hexdigest() != TABLE_SHA256:
        raise ValueError("Preflop equity table differs from its versioned contract")
    payload = json.loads(data)
    if (payload['id'] != TABLE_ID or payload['samples'] != SAMPLES
            or payload['sampling_id'] != SAMPLING_ID or len(payload['values_float32']) != 169):
        raise ValueError("Invalid preflop equity table metadata")
    return {tuple(map(int,key.split(','))): tuple(values)
            for key,values in payload['values_float32'].items()}


def preflop_equity(hole):
    # Match HULHE's own-card-first canonical key exactly, including suit ties.
    signatures = [tuple(int(4*r+s in hole) for r in range(13)) for s in range(4)]
    order = sorted(range(4), key=lambda s: signatures[s], reverse=True)
    key = tuple(sorted(4*(c//4) + order.index(c % 4) for c in hole))
    return _equity(key)


@lru_cache(maxsize=169)
def _equity(hole):
    # Bundled estimates remove first-use sampling from training and API play.
    return _table()[hole]


def _estimate(hole):
    """Offline regeneration/reference only; never used by the encoder."""
    from .strategic_features import rank_cards

    seed = int.from_bytes(blake2b(bytes(hole) + SAMPLING_ID.encode(), digest_size=16).digest(), "little")
    rng = np.random.Generator(np.random.PCG64(seed))
    decks = np.tile(np.asarray([c for c in range(52) if c not in hole], dtype=np.int16), (SAMPLES, 1))
    rows = np.arange(SAMPLES)
    for i in range(7):
        j = rng.integers(i, 50, size=SAMPLES)
        decks[rows, i], decks[rows, j] = decks[rows, j].copy(), decks[rows, i].copy()
    own = rank_cards(np.column_stack((np.tile(hole, (SAMPLES, 1)), decks[:, 2:7])))[0]
    other = rank_cards(decks[:, :7])[0]
    pay = (np.sign(own - other) + 1) / 2
    return float(pay.mean()), float(pay.std(ddof=1) / np.sqrt(SAMPLES))
