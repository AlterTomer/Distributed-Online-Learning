"""The paired-comparison instrument.

Expected values are worked by hand or taken from printed t tables, never from the
library under test -- a test that asks scipy what scipy says checks nothing.
"""

from __future__ import annotations

import math

import pytest

from dekf_bench.metrics.paired import Paired, differences, holm, paired, summarise

#: t_{0.975, 4} and t_{0.95, 4} from a printed table.
T_975_4 = 2.776445
T_95_4 = 2.131847


def test_a_textbook_paired_difference() -> None:
    """d = 1..5: mean 3, sd sqrt(2.5), se sqrt(0.5), t = 3 / sqrt(0.5)."""
    result = summarise([1.0, 2.0, 3.0, 4.0, 5.0])
    assert result.n == 5 and result.df == 4
    assert result.mean == pytest.approx(3.0)
    assert result.sd == pytest.approx(math.sqrt(2.5))
    assert result.t == pytest.approx(3.0 / math.sqrt(0.5))
    lo, hi = result.ci(0.95)
    assert lo == pytest.approx(3.0 - T_975_4 * math.sqrt(0.5), abs=1e-5)
    assert hi == pytest.approx(3.0 + T_975_4 * math.sqrt(0.5), abs=1e-5)


def test_the_critical_value_sits_at_p_005() -> None:
    """A difference exactly at t = 2.776 on 4 df is exactly at the 5% boundary."""
    se = 0.01
    at_boundary = Paired(mean=T_975_4 * se, sd=se * math.sqrt(5), n=5)
    assert at_boundary.p == pytest.approx(0.05, abs=1e-5)


def test_the_sign_is_part_of_the_result() -> None:
    a = {0: 0.30, 1: 0.32, 2: 0.31, 3: 0.33, 4: 0.30}
    b = {0: 0.20, 1: 0.23, 2: 0.20, 3: 0.22, 4: 0.21}
    forward, backward = paired(a, b), paired(b, a)
    assert forward.t > 0 > backward.t
    assert forward.t == pytest.approx(-backward.t)
    assert forward.p == pytest.approx(backward.p)


def test_only_shared_seeds_are_paired() -> None:
    a = {s: float(s) for s in range(5)}
    b = {0: 0.0, 1: 0.5, 2: 1.0}
    assert differences(a, b) == {0: 0.0, 1: 0.5, 2: 1.0}
    assert paired(a, b).n == 3


def test_one_seed_has_no_standard_error() -> None:
    result = paired({0: 0.3}, {0: 0.2})
    assert result.n == 1
    assert math.isnan(result.t) and math.isnan(result.p)
    assert result.equivalent(0.01) is None


def test_identical_differences_are_an_exact_effect() -> None:
    result = summarise([0.01] * 5)
    assert result.t == math.inf and result.p == 0.0
    assert summarise([-0.01] * 5).t == -math.inf
    assert summarise([0.0] * 5).p == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# equivalence
# --------------------------------------------------------------------------- #


def test_a_non_significant_difference_is_not_a_match() -> None:
    """The mistake the P5.7 gate made: noisy seeds fail to reject, and that was
    read as 'mean-matched'. TOST refuses to call it equal."""
    noisy = summarise([0.02, -0.02, 0.01, -0.01, 0.005])
    assert noisy.p > 0.05
    assert noisy.equivalent(margin=0.005) is False


def test_a_tight_difference_inside_the_margin_is_a_match() -> None:
    tight = summarise([0.001, -0.001, 0.0005, -0.0005, 0.0])
    assert tight.equivalent(margin=0.005) is True
    assert tight.equivalent(margin=0.0005) is False


def test_tost_is_the_90_percent_interval_inside_the_margin() -> None:
    """TOST at alpha is equivalent to the (1 - 2 alpha) interval lying inside the
    margin; the two formulations must agree on every case."""
    for values in ([0.004, 0.006, 0.005, 0.003, 0.007], [0.001, -0.002, 0.0, 0.002, -0.001],
                   [0.01, -0.01, 0.0, 0.005, -0.005]):
        result = summarise(values)
        lo, hi = result.ci(0.90)
        for margin in (0.002, 0.005, 0.01, 0.02):
            assert result.equivalent(margin) == (-margin < lo and hi < margin)
    # and the 90% interval uses t_{0.95}, not t_{0.975}
    lo, hi = summarise([1.0, 2.0, 3.0, 4.0, 5.0]).ci(0.90)
    assert hi - 3.0 == pytest.approx(T_95_4 * math.sqrt(0.5), abs=1e-5)


def test_a_margin_must_be_positive() -> None:
    with pytest.raises(ValueError, match="margin"):
        summarise([0.1, 0.2]).tost_p(0.0)


# --------------------------------------------------------------------------- #
# Holm
# --------------------------------------------------------------------------- #


def test_holm_by_hand() -> None:
    """Sorted 0.005, 0.01, 0.03, 0.04 at m = 4 -> 0.02, 0.03, 0.06, then
    max(0.06, 1 x 0.04) = 0.06, returned in the order given."""
    assert holm([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02])


def test_holm_never_lowers_a_p_value_and_caps_at_one() -> None:
    raw = [0.2, 0.5, 0.01, 0.9]
    adjusted = holm(raw)
    assert all(adj >= p for adj, p in zip(adjusted, raw, strict=True))
    assert max(adjusted) <= 1.0


def test_a_test_never_run_spends_no_alpha() -> None:
    adjusted = holm([0.01, math.nan, 0.04])
    assert adjusted[0] == pytest.approx(0.02)   # m = 2, not 3
    assert math.isnan(adjusted[1])
    assert adjusted[2] == pytest.approx(0.04)
