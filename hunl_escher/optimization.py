"""Deterministic replay-fill schedules for persistent control learners."""

import time


def replay_scaled_updates(rows, capacity, minimum, maximum, batch_size):
    """Use occupied rows, never lifetime insertions; preserve the batch gate."""
    if maximum <= 0 or rows < max(1, batch_size):
        return 0
    # Integer ceiling avoids floating-point boundary differences on resume.
    scaled = (maximum * min(rows, capacity) + capacity - 1) // capacity
    return min(maximum, max(minimum, scaled))


class ReplayScaledFitMixin:
    """The schedule has no clock/counter state beyond checkpointed replay."""

    minimum_train_steps: int

    def train_model(self, *args, **kwargs):
        maximum = self.train_steps
        rows = len(self.buffer)
        updates = replay_scaled_updates(
            rows, self.buffer_size, self.minimum_train_steps, maximum, self.batch_size
        )
        started = time.perf_counter()
        if hasattr(self, "last_target_cache_stats"):
            self.last_target_cache_stats = {}
        self.last_fit_stats = {"rows": rows, "updates": 0, "loss": None,
                               "scheduled_updates": updates, "maximum_updates": maximum}
        self.train_steps = updates
        try:
            loss = super().train_model(*args, **kwargs)
            self.last_fit_stats.update(loss=loss, updates=updates if loss is not None else 0)
            return loss
        finally:
            self.train_steps = maximum
            self.last_fit_stats["seconds"] = time.perf_counter() - started
