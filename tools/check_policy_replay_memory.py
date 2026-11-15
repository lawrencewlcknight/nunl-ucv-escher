"""Bounded synthetic grouping check; no learning, credentials or network access.

Run each measurement in a fresh process. Default input is 100k unique rows,
using the production feature width. Use --reference to measure the old dense
grouping separately. This is not a full training-VM peak-memory measurement.
"""
import argparse
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hunl_escher.features import make_feature_encoder
from unbiased_escher.efficient_replay import CompactReservoirBuffer
from unbiased_escher.replay_grouping import CHUNK_ROWS, grouped_policy_data


def peak_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=100000)
    parser.add_argument("--reference", action="store_true")
    args = parser.parse_args()
    if not 0 < args.rows <= 2**24:
        parser.error("rows must be in 1..2^24 for exact synthetic float32 IDs")
    encoder = make_feature_encoder()
    actions = 7
    buffer = CompactReservoirBuffer(args.rows, encoder.policy_size, actions)
    player_columns, street_columns = encoder.replay_context_columns()
    for start in range(0, args.rows, CHUNK_ROWS):
        stop = min(start + CHUNK_ROWS, args.rows)
        ids = np.arange(start, stop)
        buffer.infostate_buf[start:stop] = 0
        buffer.infostate_buf[start:stop, 0] = ids
        buffer.infostate_buf[start:stop, player_columns] = np.eye(2)[ids % 2]
        buffer.infostate_buf[start:stop, street_columns] = np.eye(4)[ids % 4]
        buffer.q_value_buf[start:stop] = 1 / actions
        buffer.q_value_mask_buf[start:stop] = 1
        buffer.iteration_buf[start:stop, 0] = 1 + ids % 100
    buffer.cur_id = args.rows
    before = peak_bytes()
    started = time.perf_counter()
    if args.reference:
        # Original wide-matrix np.unique path, including its weighted targets.
        features = buffer.infostate_buf
        policies = buffer.q_value_buf.astype(np.float64)
        weights = (buffer.iteration_buf.astype(np.float64).ravel() / 100 * 2) ** 2
        unique, first, inverse = np.unique(features, axis=0, return_index=True, return_inverse=True)
        masses = np.zeros(len(unique))
        numerator = np.zeros((len(unique), actions))
        np.add.at(masses, inverse, weights)
        np.add.at(numerator, inverse, policies * weights[:, None])
        targets = numerator / masses[:, None]
        masks = buffer.q_value_mask_buf[first]
        objective_weights = masses * len(unique) / len(features)
        assert np.isfinite(targets).all() and masks.shape == targets.shape
        assert np.isfinite(objective_weights).all()
        groups = len(unique)
    else:
        data, stats = grouped_policy_data(buffer, 100, 2, "cpu",
            context_columns=encoder.replay_context_columns())
        groups = len(data[0])
        assert stats["full_feature_copy_bytes"] == 0
        assert np.shares_memory(data[0].source, buffer.infostate_buf)
        assert data[0][:16].shape == (min(args.rows,16), encoder.policy_size)
    assert groups == args.rows
    after = peak_bytes()
    print(json.dumps({"mode": "dense_reference" if args.reference else "indexed",
        "rows": args.rows, "groups": groups, "feature_width": encoder.policy_size,
        "replay_bytes": buffer.nbytes(), "feature_matrix_bytes": buffer.infostate_buf.nbytes,
        "peak_before_grouping_bytes": before, "peak_after_grouping_bytes": after,
        "incremental_peak_bytes": max(0, after - before),
        "grouping_seconds": time.perf_counter() - started}))


if __name__ == "__main__":
    main()
