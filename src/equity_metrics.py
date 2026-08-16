"""
Performance metrics computed FROM AN EQUITY CURVE ``ce`` (monthly observations).

Annualised return/volatility, Sharpe, max drawdown and Calmar. Kept for the
parts of the analysis that work directly on an equity curve rather than on a
return series (see ``metrics`` for the return-series variant).

Moved here from ``Code/performance_metrics.py`` during the src/ refactor;
the public function names are unchanged.
"""

import numpy as np
import pandas as pd


def ann_ret(ce):
    # Compute annualized Returns starting from equity curve
    t = len(ce) / 12
    ret_ann = (ce.values[-1] / ce.values[0]) ** (1 / t) - 1
    return ret_ann


def ann_vol(ce_returns):
    # Compute annualized volatility starting from equity curve returns
    vol_ann = ce_returns.std() * np.sqrt(12)
    return vol_ann


def sharpe_ratio(ce):
    # Compute Sharpe Ratio starting from equity curve
    ce_returns = ce.pct_change(1).fillna(0)
    ret_ann = ann_ret(ce)
    vol_ann = ann_vol(ce_returns)
    sharpe_ratio = ret_ann / vol_ann
    return sharpe_ratio


def max_drawdown(ce):
    # Compute Max Drawdown starting from equity curve returns
    t_minus_max = [(ce.values[i] / ce.values[0:i].max()) - 1 for i in range(1, len(ce))]
    try:
        max_drawdown = np.min(t_minus_max)
    except ValueError:
        t_minus_max = [(ce.values[-1] / ce.values[-1].max())[0] - 1]
        max_drawdown = np.min(t_minus_max)
    return max_drawdown


def calmar_ratio(ce):
    # Compute Calmar Ratio starting from equity curve returns
    drawdown = max_drawdown(ce)
    aret = ann_ret(ce)
    calmar = aret / np.abs(drawdown)
    return calmar


def get_performance_metrics(ce):
    # Compute Performance Metrics
    aret = ann_ret(ce) * 100
    avol = ann_vol(ce.pct_change().fillna(0)) * 100
    sr = sharpe_ratio(ce)
    mxdd = max_drawdown(ce) * 100
    cr = calmar_ratio(ce)
    df_perf = pd.DataFrame([aret, avol, sr, mxdd, cr],
                           index=['AnnRet', 'AnnVol', 'Sharpe', 'MaxDD', 'Calmar'],
                           columns=['Metrics'])
    return df_perf
