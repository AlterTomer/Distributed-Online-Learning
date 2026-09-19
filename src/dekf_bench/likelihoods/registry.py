r"""Name to likelihood, so a script never constructs one directly.

Nine call sites hardcoded ``Categorical(config.model.output_dim)`` while
`gaussian.py` sat built, tested and unreachable from any config. That path is the
one that matters: ``score`` and ``innovation`` coincide under a softmax and
differ by $\sigma^{-2}$ under a Gaussian (D60), which is how the filter's mean
update stayed wrong by that factor until a linear-Gaussian test caught it. It is
also what `ex:linear` needs -- a linear probe with Gaussian observations makes
the \ac{ekf} an exact Kalman filter (IMPLEMENTATION.md section 13.9).
"""

from __future__ import annotations

from typing import Any

from dekf_bench.likelihoods.base import LikelihoodError
from dekf_bench.likelihoods.categorical import Categorical
from dekf_bench.likelihoods.gaussian import Gaussian

#: Every likelihood a config may name.
LIKELIHOODS = {"categorical": Categorical, "gaussian": Gaussian}


def likelihood_names() -> list[str]:
    return sorted(LIKELIHOODS)


def build_likelihood(config: Any) -> Any:
    """The likelihood a config asks for, sized to the model's output.

    ``output_dim`` is read from the model rather than named twice, since a config
    that can disagree with itself about $q$ eventually does.
    """
    name = getattr(config.model, "likelihood", "categorical")
    if name not in LIKELIHOODS:
        raise LikelihoodError(f"unknown likelihood {name!r}; available: {likelihood_names()}")
    if name == "gaussian":
        per_position = getattr(config.model, "observation_variances", None) or None
        return Gaussian(
            output_dim=config.model.output_dim,
            variance=getattr(config.model, "observation_variance", 1.0),
            variances=None if per_position is None else tuple(float(v) for v in per_position),
        )
    return Categorical(config.model.output_dim)
