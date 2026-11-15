"""Exact replay grouping without copying the full feature matrix.

Scratch space scales with row indices and action targets, not rows * features.
Fingerprints accelerate grouping; exact comparisons resolve even hash collisions.
The reservoir is read-only for the lifetime of a grouped fit.
"""
from hashlib import blake2b
import time

import numpy as np
import torch


CHUNK_ROWS = 4096


def reservoir_counts(buffer):
    seen = int(buffer.cur_id)
    capacity = int(buffer.buffer_size)
    retained = min(seen, capacity)
    allocated = (buffer.nbytes() if hasattr(buffer, "nbytes") else
                 sum(getattr(buffer, name).nbytes for name in (
                     "infostate_buf", "q_value_buf", "q_value_mask_buf", "iteration_buf")))
    return {"capacity": capacity, "seen": seen, "retained": retained,
            "not_retained": seen - retained,
            "retained_fraction": retained / seen if seen else None,
            "allocated_bytes": int(allocated)}


def retained_context_counts(buffer, columns):
    counts = np.zeros((2, 4), dtype=np.int64)
    size = min(int(buffer.cur_id), int(buffer.buffer_size))
    for start in range(0, size, CHUNK_ROWS):
        rows = slice(start, min(start + CHUNK_ROWS, size))
        players = np.argmax(buffer.infostate_buf[rows, columns[0]], axis=1)
        streets = np.argmax(buffer.infostate_buf[rows, columns[1]], axis=1)
        np.add.at(counts, (players, streets), 1)
    return counts.tolist()


def learner_replay_allocation(solver):
    """Allocated dense replay only: excludes models, workers and fit scratch."""
    calibration = solver.calibration_trainer
    sizes = {"regret": sum(t.buffer.nbytes() for t in solver.regret_trainers),
             "policy": solver.ave_policy_trainer.buffer.nbytes(),
             "critic": sum(t.buffer.nbytes() for t in solver.q_value_trainer.members),
             "calibration": (calibration.buffer.features.nbytes + calibration.buffer.targets.nbytes
                             if calibration is not None else 0)}
    return {**{key: int(value) for key, value in sizes.items()},
            "total": int(sum(sizes.values()))}


class IndexedFeatureRows:
    """Gather only the requested minibatch from the original CPU reservoir."""

    def __init__(self, source, rows, device):
        self.source = source
        self.rows = rows
        self.device = torch.device(device)
        self.shape = (len(rows), source.shape[1])

    def __len__(self):
        return len(self.rows)

    def numpy_batch(self, start, stop):
        return self.source[self.rows[start:stop]]

    def __getitem__(self, indices):
        if isinstance(indices, torch.Tensor):
            indices = indices.detach().cpu().numpy()
        return torch.as_tensor(self.source[self.rows[indices]], device=self.device)

    def index_select(self, dim, indices):
        if dim != 0:
            raise ValueError("Indexed replay supports row selection only")
        return self[indices]

    def full_batch(self):
        # Only explicit full-batch callers (e.g. tiny engineering tests).
        return self[:]


def _fingerprints(features, chunk_rows):
    fingerprints = np.empty(len(features), dtype="V16")
    for start in range(0, len(features), chunk_rows):
        stop = min(start + chunk_rows, len(features))
        chunk = features[start:stop].copy()
        if not np.isfinite(chunk).all():
            raise ValueError("Nonfinite policy replay features")
        chunk[chunk == 0] = 0  # Numeric equality treats -0 and +0 identically.
        fingerprints[start:stop] = [
            blake2b(row, digest_size=16, person=b"holdem-group-v1").digest()
            for row in chunk
        ]
    return fingerprints


def exact_group_indices(features, chunk_rows=CHUNK_ROWS):
    """Return first/inverse indexes; deterministic order, RNG-free, collision-safe."""
    if chunk_rows < 1 or len(features) == 0:
        raise ValueError("Grouping requires rows and a positive chunk size")
    hashes = _fingerprints(features, chunk_rows)
    order = np.argsort(hashes, kind="stable")
    ordered_hashes = hashes[order]
    starts = np.r_[0, np.flatnonzero(ordered_hashes[1:] != ordered_hashes[:-1]) + 1]
    del hashes, ordered_hashes
    # Check equality within each fingerprint bucket in bounded feature batches.
    # Most distinct poker states have different hashes and need no comparison.
    boundaries = np.zeros(len(features), dtype=bool)
    boundaries[starts] = True
    collision_buckets = set()
    for start in range(1, len(features), chunk_rows):
        stop = min(start + chunk_rows, len(features))
        positions = np.flatnonzero(~boundaries[start:stop]) + start
        different = np.any(features[order[positions]] != features[order[positions - 1]], axis=1)
        if different.any():
            collision_buckets.update(np.searchsorted(starts, positions[different], side="right") - 1)
    # Rare collision fallback: stable numeric lexicographic sorting using ONE
    # column at a time. Even an all-colliding input never copies the wide matrix.
    for bucket in sorted(collision_buckets):
        start = starts[bucket]
        stop = starts[bucket + 1] if bucket + 1 < len(starts) else len(order)
        rows = order[start:stop].copy()
        for column in range(features.shape[1] - 1, -1, -1):
            rows = rows[np.argsort(features[rows, column], kind="stable")]
        order[start:stop] = rows
        for offset in range(start + 1, stop, chunk_rows):
            end = min(offset + chunk_rows, stop)
            boundaries[offset:end] = np.any(
                features[order[offset:end]] != features[order[offset - 1:end - 1]], axis=1)
    first = order[boundaries].copy()
    inverse = np.empty(len(features), dtype=np.int64)
    inverse[order] = np.cumsum(boundaries, dtype=np.int64) - 1
    return first, inverse


def grouped_policy_data(buffer, iteration, gamma, device, *, context_columns=None,
                        chunk_rows=CHUNK_ROWS):
    """Keep the original weighted objective and reservoir-order target summation."""
    started = time.perf_counter()
    counts = reservoir_counts(buffer)
    size = counts["retained"]
    if size <= 0:
        raise ValueError("Cannot fit an average policy from an empty reservoir")
    if not np.isfinite(iteration) or iteration <= 0:
        raise ValueError("Policy iteration must be positive and finite")
    # Production replay already is float32: this is a view, not a wide copy.
    # Preserve the legacy parent trainer's float32 grouping semantics as well.
    features = np.asarray(buffer.infostate_buf[:size], dtype=np.float32)
    feature_copy_bytes = 0 if np.shares_memory(features, buffer.infostate_buf) else features.nbytes
    first, inverse = exact_group_indices(features, chunk_rows)
    groups = len(first)
    masses = np.zeros(groups, dtype=np.float64)
    numerators = np.zeros((groups, buffer.action_size), dtype=np.float64)
    masks = np.empty((groups, buffer.action_size), dtype=np.float32)
    retained_counts = np.zeros((2, 4), dtype=np.int64)
    unique_counts = np.zeros((2, 4), dtype=np.int64)
    weight_sums = np.zeros((2, 4), dtype=np.float64)
    squared_weight_sums = np.zeros((2, 4), dtype=np.float64)

    def cells(rows):
        player_columns, street_columns = context_columns
        return (np.argmax(features[rows, player_columns], axis=1),
                np.argmax(features[rows, street_columns], axis=1))

    for start in range(0, groups, chunk_rows):
        stop = min(start + chunk_rows, groups)
        rows = first[start:stop]
        masks[start:stop] = buffer.q_value_mask_buf[rows]
        if context_columns is not None:
            np.add.at(unique_counts, cells(rows), 1)
    for start in range(0, size, chunk_rows):
        stop = min(start + chunk_rows, size)
        rows = slice(start, stop)
        ids = inverse[rows]
        weights = np.power(np.maximum(buffer.iteration_buf[rows, 0].astype(np.float64), 0)
                           / float(iteration) * 2, gamma)
        values = buffer.q_value_buf[rows].astype(np.float64)
        if not np.isfinite(weights).all() or not np.isfinite(values).all():
            raise ValueError("Nonfinite policy replay weights or targets")
        if not np.array_equal(buffer.q_value_mask_buf[rows], masks[ids]):
            raise ValueError("Legal-action masks differ within an information set")
        np.add.at(masses, ids, weights)
        np.add.at(numerators, ids, values * weights[:, None])
        if context_columns is not None:
            cell = cells(rows)
            np.add.at(retained_counts, cell, 1)
            np.add.at(weight_sums, cell, weights)
            np.add.at(squared_weight_sums, cell, weights * weights)
    if np.any(masses <= 0):
        raise ValueError("Grouped average-policy target has non-positive mass")
    del inverse
    targets = np.empty_like(masks)
    for start in range(0, groups, chunk_rows):
        stop = min(start + chunk_rows, groups)
        targets[start:stop] = numerators[start:stop] / masses[start:stop, None]
    del numerators
    weights = (masses * (float(groups) / size)).astype(np.float32)
    summary = {**counts, "fit_iteration": int(iteration), "unique_groups": groups,
               "unique_fraction": groups / size, "group_order": "exact_fingerprint_v1",
               "chunk_rows": chunk_rows, "full_feature_copy_bytes": int(feature_copy_bytes),
               "grouped_index_bytes": int(first.nbytes),
               "grouped_target_bytes": int(targets.nbytes + masks.nbytes + weights.nbytes),
               "grouping_seconds": time.perf_counter() - started}
    if context_columns is not None:
        ess = np.divide(weight_sums ** 2, squared_weight_sums,
                        out=np.zeros((2, 4)), where=squared_weight_sums > 0)
        summary.update(retained_by_player_street=retained_counts.tolist(),
                       unique_by_player_street=unique_counts.tolist(),
                       weight_mass_by_player_street=weight_sums.tolist(),
                       effective_rows_by_player_street=ess.tolist())
    tensors = tuple(torch.as_tensor(value, device=device) for value in (targets, masks, weights))
    return (IndexedFeatureRows(features, first, device), *tensors), summary
