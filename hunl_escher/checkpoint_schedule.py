"""Persisted six-hour wall recovery checks and active-learner policy fitting."""
from dataclasses import dataclass, asdict
import time


RECOVERY_RECENT_SECONDS = 3600


@dataclass
class CheckpointSchedule:
    # Epoch times survive restarts; downtime counts toward the next check.
    last_recovery_wall_seconds: float | None = None
    last_policy_seconds: float = 0.0
    last_recovery_check_wall_seconds: float | None = None

    def __post_init__(self):
        if self.last_recovery_check_wall_seconds is None:
            self.last_recovery_check_wall_seconds = (time.time() if self.last_recovery_wall_seconds is None
                                                     else self.last_recovery_wall_seconds)

    def due(self, *, iteration, final, active_seconds, wall_seconds,
            recovery_seconds, policy_seconds, checkpoint_every=0, policy_every=0):
        policy = (final
                  or (policy_every > 0 and iteration % policy_every == 0)
                  or active_seconds - self.last_policy_seconds >= policy_seconds)
        timed_recovery = False
        elapsed = wall_seconds - self.last_recovery_check_wall_seconds
        if recovery_seconds > 0 and elapsed >= recovery_seconds:
            # Consume this slot even when a recent save suppresses it. Advance
            # on the original grid; long iterations/downtime cause no catch-up
            # burst and incidental policy saves do not shift the six-hour grid.
            self.last_recovery_check_wall_seconds += (elapsed // recovery_seconds) * recovery_seconds
            timed_recovery = (self.last_recovery_wall_seconds is None or
                              wall_seconds - self.last_recovery_wall_seconds >= RECOVERY_RECENT_SECONDS)
        # Explicit iteration saves, policy snapshots and final output remain
        # unconditional. The recent-save rule applies to timed recovery only.
        recovery = (policy or timed_recovery
                    or (checkpoint_every > 0 and iteration % checkpoint_every == 0))
        return bool(recovery), bool(policy)

    def completed(self, active_seconds, *, wall_seconds, recovery, policy):
        if recovery:
            self.last_recovery_wall_seconds = wall_seconds
        if policy:
            self.last_policy_seconds = active_seconds

    def state(self):
        return asdict(self)
