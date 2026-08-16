"""Tests for src.metrics (return-series performance metrics)."""

import numpy as np
import pandas as pd
import pytest

from src import metrics as pm


def test_annualization_factor():
    assert pm.get_annualization_factor("d") == 252
    assert pm.get_annualization_factor("m") == 12
    with pytest.raises(ValueError):
        pm.get_annualization_factor("x")


def test_equity_curve_compounds():
    returns = pd.Series([0.1, 0.0, -0.05])
    ec = pm.build_equity_curve(returns)
    # (1.1)(1.0)(0.95) = 1.045
    assert ec.iloc[-1] == pytest.approx(1.1 * 1.0 * 0.95)


def test_constant_positive_return_matches_original_convention():
    # Documents the project's original convention: the annualised return is
    # (equity[-1] / equity[0]) ** (1 / t) - 1 with equity[0] already including the
    # first period's return and t = n / periods_per_year. For 12 months of 1%
    # this yields 1.01**11 - 1 (not 1.01**12 - 1). Preserved as-is for fidelity.
    returns = pd.Series([0.01] * 12)
    aret = pm.ann_ret_from_equity(returns, "m")
    assert aret == pytest.approx((1.01 ** 11) - 1, rel=1e-9)


def test_max_drawdown_is_negative_on_a_dip():
    returns = pd.Series([0.10, -0.30, 0.05])
    mdd = pm.max_drawdown_from_equity(returns)
    assert mdd < 0


def test_calmar_infinite_without_drawdown():
    returns = pd.Series([0.01, 0.02, 0.03])  # monotonically rising -> no drawdown
    assert np.isinf(pm.calmar_ratio_from_equity(returns, "m"))


def test_metrics_frame_shape_and_index():
    returns = pd.Series([0.01, -0.02, 0.03, 0.00, 0.015])
    perf = pm.get_performance_metrics_from_returns(returns, "m")
    assert list(perf.index) == ["AnnRet", "AnnVol", "Sharpe", "MaxDD", "Calmar"]
    assert list(perf.columns) == ["Metrics"]
