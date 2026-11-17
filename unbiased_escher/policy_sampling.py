"""Grouped-policy sampling with an unchanged expected weighted objective."""
import random

import numpy as np
import torch


class PolicyGroupSampler:
    """Uniform groups or iid objective-mass draws, with the right loss factors.

    For n training groups with weights w, the original objective is
    sum(w_i * loss_i) / n. Sampling p_i=w_i/sum(w) therefore needs the
    CONSTANT factor sum(w)/n, not w_i a second time. The training partition's
    mean weight preserves the legacy scale even when validation groups exist.
    A cached float64 CDF avoids rebuilding an O(n) distribution every update.
    """

    def __init__(self, ids, weights, batch_size, *, mode="uniform", seed=0):
        if mode not in ("uniform", "objective_mass"):
            raise ValueError("Unknown policy group sampler")
        self.ids = np.asarray(ids, dtype=np.int64)
        if not len(self.ids) or batch_size < 1:
            raise ValueError("Sampler requires training groups and a positive batch size")
        self.weights = weights
        self.batch_size = min(int(batch_size), len(self.ids))
        self.mode = mode
        self.rng = np.random.default_rng(seed)
        self.cdf = None
        if mode == "objective_mass":
            mass = weights[torch.as_tensor(self.ids, device=weights.device)].detach().cpu().numpy().astype(np.float64)
            if not np.isfinite(mass).all() or np.any(mass <= 0):
                raise ValueError("Policy group masses must be finite and positive")
            self.cdf = np.cumsum(mass)
            self.factor = weights.new_tensor(self.cdf[-1] / len(mass))
            self.cdf /= self.cdf[-1]
            self.cdf[-1] = 1.0

    def sample(self):
        if self.batch_size == len(self.ids):
            selected = self.ids  # Both modes use the exact objective for a full batch.
        elif self.mode == "uniform":
            # Keep the existing Python RNG and without-replacement behavior exactly.
            selected = self.ids[random.sample(range(len(self.ids)), self.batch_size)]
        else:
            selected = self.ids[np.searchsorted(self.cdf, self.rng.random(self.batch_size), side="right")]
        idx = torch.as_tensor(selected, device=self.weights.device)
        factors = (self.factor if self.mode == "objective_mass" and self.batch_size < len(self.ids)
                   else self.weights[idx])
        return idx, factors
