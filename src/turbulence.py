"""
Kritzman-Li financial turbulence index.

Turbulence measures how "unusual" a cross-section of asset returns is relative to
its historical behaviour, as the Mahalanobis distance of each period's return
vector from the sample mean, using the sample covariance matrix:

    d_t = (y_t - mu) * inv(Sigma) * (y_t - mu)^T

High values flag periods where returns are extreme and/or their usual
correlation structure breaks down. Used in the notebook as one of the features
that describe the market-risk regime.

Reference: Kritzman & Li, "Skulls, Financial Turbulence, and Risk Management",
Financial Analysts Journal (2010).
"""

import numpy as np
import pandas as pd


def financial_turbulence(returns: pd.DataFrame) -> pd.Series:
    """Compute the turbulence index for a DataFrame of asset returns.

    Parameters
    ----------
    returns : pd.DataFrame
        Rows are periods, columns are assets. Must contain no NaNs.

    Returns
    -------
    pd.Series
        Turbulence value per period, indexed like ``returns``.
    """
    if returns.isna().any().any():
        raise ValueError("`returns` must not contain NaN values.")

    mu = returns.mean().values
    cov = returns.cov().values
    cov_inv = np.linalg.pinv(cov)  # pseudo-inverse: robust to near-singular covariance

    centered = returns.values - mu
    # Row-wise quadratic form d_t = z_t . cov_inv . z_t  (einsum avoids a Python loop)
    distances = np.einsum("ij,jk,ik->i", centered, cov_inv, centered)

    return pd.Series(distances, index=returns.index, name="Financial turbulence")
