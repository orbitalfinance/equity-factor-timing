"""
Performance metrics computed FROM AN EQUITY CURVE ``ce`` (monthly observations).

Annualised return/volatility, Sharpe, max drawdown and Calmar.

NOT USED BY THE NOTEBOOK. This is a faithful port of the original
``Code/performance_metrics.py``, kept so the repository still carries the code
as it was written for the Project Work. The analysis imports ``metrics``
instead (as ``pm``) and works from return series throughout.

It differs from ``metrics`` in three ways, all deliberate; please do not
"fix" them without a reason:

- ``max_drawdown`` takes the running peak over ``ce[0:i]``, excluding the
  current point, where ``metrics`` uses ``cummax()``, which includes it. The
  two agree on any curve that declines at least once — at a trough the peak is
  always reached strictly earlier, so the excluded point never was the max.
  They part only on a never-declining curve, where this version reports a
  positive "drawdown" and ``metrics`` reports zero.
- Annualisation is fixed at 12: this module is monthly-only by design, while
  ``metrics`` takes a ``freq`` argument.
- ``calmar_ratio`` has no zero-drawdown guard (``metrics`` returns ``inf``).
  Unreachable in practice, for the same reason as above.

The ``except ValueError`` branch in ``max_drawdown`` catches the empty-list
case of a single-observation curve. It is written for a DataFrame ``ce`` and
would raise ``IndexError`` on a Series; nothing currently calls it.
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
