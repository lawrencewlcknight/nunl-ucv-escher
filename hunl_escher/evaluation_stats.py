"""Fixed-sample tests. Pilots choose sample sizes, never significance stopping."""
import math

import numpy as np
from scipy.stats import norm, t


def values(data):
    x = np.asarray(data, dtype=np.float64)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("At least two finite independent observations are required")
    return x


def sample_plan(pilot, *, other=None, minimum=100000, maximum=1000000,
                effect_bb100=10., alpha=.05, power=.8):
    """Observations are pair-average bb locally, per-hand bb remotely.

    For two independent arms returns observations PER ARM. Normal-approximation
    power is a planning estimate, not a guarantee based on estimated variance.
    """
    if not (2 <= minimum <= maximum and math.isfinite(effect_bb100) and
            effect_bb100 > 0 and 0 < alpha < 1 and .5 < power < 1):
        raise ValueError("Invalid statistical design")
    variance = float(values(pilot).var(ddof=1))
    if other is not None:
        variance += float(values(other).var(ddof=1))
    factor = float(norm.ppf(1 - alpha / 2) + norm.ppf(power))
    required = max(2, math.ceil(factor**2 * variance / (effect_bb100 / 100)**2))
    selected = min(maximum, max(minimum, required))
    return dict(required=required, selected=selected, capped=required > maximum,
                target_effect_bb100=effect_bb100, alpha=alpha, power=power,
                estimated_detectable_effect_bb100=100 * factor * math.sqrt(variance / selected),
                pilot_variance_bb2=variance, pilot_excluded=True)


def estimate(data, *, other=None, alpha=.05, complete=True):
    """Student interval for independent pairs, Welch interval for remote arms."""
    x = values(data)
    mean, v = float(x.mean()), float(x.var(ddof=1) / len(x))
    dof = len(x) - 1
    sizes = [len(x)]
    if other is not None:
        y = values(other)
        vy = float(y.var(ddof=1) / len(y))
        mean -= float(y.mean())
        denominator = v*v / (len(x)-1) + vy*vy / (len(y)-1)
        dof = (v+vy)**2 / denominator if denominator else len(x)+len(y)-2
        v += vy
        sizes.append(len(y))
    half = float(t.ppf(1-alpha/2, dof)) * math.sqrt(v)
    low, high = 100*(mean-half), 100*(mean+half)
    # Zero observed variance does not establish a zero-uncertainty population.
    usable = complete and v > 0
    conclusion = ("improved" if low > 0 else "regressed" if high < 0 else "inconclusive") if usable else "incomplete_or_degenerate"
    return dict(bb_per_100=100*mean, interval=[low, high], confidence=1-alpha,
                observations=sizes, standard_error_bb100=100*math.sqrt(v),
                confirmatory=usable, conclusion=conclusion)
