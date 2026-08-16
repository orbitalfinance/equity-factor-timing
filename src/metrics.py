"""
Performance metrics computed FROM A RETURN SERIES.

Annualised return/volatility, Sharpe, max drawdown and Calmar, rebuilding the
equity curve from returns. The notebook imports this module as ``pm`` and calls
``get_performance_metrics_from_returns(returns, freq)``.

``freq`` is one of 'd' | 'w' | 'm' | 'y' (sets the annualisation factor).

Moved here from ``Code/performance_metrics_ret.py`` during the src/ refactor;
the public function names are unchanged for backward compatibility.
"""

import numpy as np
import pandas as pd


# Rebuild equity curve from returns
def build_equity_curve(returns):
    equity_curve = (1 + returns).cumprod()
    return equity_curve


# Function to determine the annualization factor
def get_annualization_factor(freq):
    if freq == 'd':
        return 252  # Daily data: 252 trading days in a year
    elif freq == 'w':
        return 52  # Weekly data: 52 weeks in a year
    elif freq == 'm':
        return 12  # Monthly data: 12 months in a year
    elif freq == 'y':
        return 1  # Yearly data: 1 year
    else:
        raise ValueError("Invalid frequency. Use 'd', 'w', 'm', or 'y'.")


# Annualized Return from the rebuilt equity curve
def ann_ret_from_equity(returns, freq):
    equity_curve = build_equity_curve(returns)
    periods_per_year = get_annualization_factor(freq)
    t = len(equity_curve) / periods_per_year
    ret_ann = (equity_curve.values[-1] / equity_curve.values[0]) ** (1 / t) - 1
    return ret_ann


# Annualized Volatility from returns
def ann_vol(returns, freq):
    periods_per_year = get_annualization_factor(freq)
    vol_ann = returns.std() * np.sqrt(periods_per_year)
    return vol_ann


# Sharpe Ratio from the rebuilt equity curve
def sharpe_ratio_from_equity(returns, freq):
    equity_curve = build_equity_curve(returns)
    equity_returns = equity_curve.pct_change(1).fillna(0)
    ret_ann = ann_ret_from_equity(returns, freq)
    vol_ann = ann_vol(equity_returns, freq)
    sharpe_ratio = ret_ann / vol_ann
    return sharpe_ratio


# Max Drawdown from the rebuilt equity curve
def max_drawdown_from_equity(returns):
    equity_curve = build_equity_curve(returns)
    drawdowns = equity_curve / equity_curve.cummax() - 1
    max_drawdown = drawdowns.min()
    return max_drawdown


# Calmar Ratio from the rebuilt equity curve
def calmar_ratio_from_equity(returns, freq):
    drawdown = max_drawdown_from_equity(returns)
    aret = ann_ret_from_equity(returns, freq)
    if drawdown == 0:
        return np.inf
    calmar = aret / np.abs(drawdown)
    return calmar


# Compute all performance metrics from returns (simulating equity curve)
def get_performance_metrics_from_returns(returns, freq):
    aret = ann_ret_from_equity(returns, freq) * 100
    avol = ann_vol(returns, freq) * 100
    sr = sharpe_ratio_from_equity(returns, freq)
    mxdd = max_drawdown_from_equity(returns) * 100
    cr = calmar_ratio_from_equity(returns, freq)

    df_perf = pd.DataFrame({
        'Metrics': [aret, avol, sr, mxdd, cr]
    }, index=['AnnRet', 'AnnVol', 'Sharpe', 'MaxDD', 'Calmar'])

    return df_perf
