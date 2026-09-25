"""Paired comparisons across seeds -- the one inferential instrument the reports use.

Every t quoted in this project is a paired test on per-seed differences. Two cells
run under the same seed share the graph, the data stream and theta_0 (D8), so
d_s = a_s - b_s cancels everything the seeds have in common, and the unit of
observation is the seed: a run's time steps are averaged into one settled value
first, so their autocorrelation never inflates the degrees of freedom. The null is
always mu_d = 0, two-sided, on n - 1 degrees of freedom (4 at five seeds, where the
5% critical value is 2.776 and not 2.0).

Three mistakes this module exists to stop:

* **A non-rejection is not a match.** |t| below the critical value is absence of
  evidence. A claim that two cells are *the same* needs an equivalence test against
  a margin stated in advance: TOST rejects both one-sided nulls mu_d <= -margin and
  mu_d >= +margin, which is the same as the (1 - 2 alpha) interval lying inside
  (-margin, +margin). At five seeds a real mismatch passes a plain t-test easily.
* **A table is a family.** Ten rows at 5% each expect half a false positive by
  chance alone. `holm` adjusts a table's p-values for family-wise error; it is
  uniformly more powerful than Bonferroni and valid under any dependence between
  rows -- which matters here, because rows share seeds.
* **The sign is part of the result.** `Paired.t` is signed. A reader should never
  have to recover the direction of an effect from a separate column.

No normality check is possible at n = 5, and none is attempted. Nor is there a
nonparametric fallback: a sign-flip permutation test has 2**5 = 32 arrangements, so
its smallest two-sided p is 2/32 = 0.0625 and it can never reject at 5%. The t-test
is the only test that can, and it buys that from the normality assumption.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from scipy import stats


@dataclass(frozen=True)
class Paired:
    """A paired difference summarised: its mean, spread and seed count.

    Everything inferential is derived, so nothing can drift out of step with the
    numbers it was computed from.
    """

    mean: float
    sd: float
    n: int

    @property
    def df(self) -> int:
        return self.n - 1

    @property
    def se(self) -> float:
        return self.sd / math.sqrt(self.n) if self.n >= 2 else math.nan

    @property
    def t(self) -> float:
        """Signed. NaN below two seeds: one seed has no standard error."""
        if self.n < 2:
            return math.nan
        if self.se == 0:
            # Every seed gave the same difference: an exact effect, or exactly none.
            return math.copysign(math.inf, self.mean) if self.mean else 0.0
        return self.mean / self.se

    @property
    def p(self) -> float:
        """Two-sided p-value against mu_d = 0."""
        if self.n < 2:
            return math.nan
        return float(2.0 * stats.t.sf(abs(self.t), self.df))

    def ci(self, level: float = 0.95) -> tuple[float, float]:
        """The two-sided `level` interval for mu_d."""
        if self.n < 2:
            return (math.nan, math.nan)
        half = float(stats.t.ppf(0.5 + level / 2.0, self.df)) * self.se
        return (self.mean - half, self.mean + half)

    def tost_p(self, margin: float) -> float:
        """p for equivalence within +/- `margin`: the larger of the two one-sided p."""
        if margin <= 0:
            raise ValueError(f"an equivalence margin must be > 0, got {margin}")
        if self.n < 2:
            return math.nan
        if self.se == 0:
            return 0.0 if abs(self.mean) < margin else 1.0
        below = stats.t.sf((self.mean + margin) / self.se, self.df)  # H0: mu <= -margin
        above = stats.t.cdf((self.mean - margin) / self.se, self.df)  # H0: mu >= +margin
        return float(max(below, above))

    def equivalent(self, margin: float, alpha: float = 0.05) -> bool | None:
        """True if shown equal within the margin, False if not shown; None below two seeds."""
        p = self.tost_p(margin)
        return None if math.isnan(p) else p < alpha


def differences(a: Mapping[int, float], b: Mapping[int, float]) -> dict[int, float]:
    """a - b per seed, over the seeds both have. Unpaired seeds are dropped, never imputed."""
    return {seed: a[seed] - b[seed] for seed in sorted(set(a) & set(b))}


def summarise(values: Mapping[int, float] | Sequence[float]) -> Paired:
    """A `Paired` from differences already taken -- the entry point for a
    difference of differences, which is itself one number per seed (D54)."""
    xs = list(values.values()) if isinstance(values, Mapping) else list(values)
    n = len(xs)
    if n == 0:
        return Paired(mean=math.nan, sd=math.nan, n=0)
    mean = sum(xs) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in xs) / (n - 1)) if n >= 2 else math.nan
    return Paired(mean=mean, sd=sd, n=n)


def paired(a: Mapping[int, float], b: Mapping[int, float]) -> Paired:
    """The paired comparison of a against b, seed by seed."""
    return summarise(differences(a, b))


def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values, in the order given.

    NaNs -- rows with fewer than two seeds -- are passed through and do not count
    toward the family size: a test that was never run cannot spend alpha.
    """
    live = sorted((p, i) for i, p in enumerate(pvalues) if not math.isnan(p))
    m = len(live)
    adjusted = [math.nan] * len(pvalues)
    running = 0.0
    for rank, (p, index) in enumerate(live):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[index] = running
    return adjusted
