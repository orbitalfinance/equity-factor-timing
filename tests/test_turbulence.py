"""Tests for src.turbulence (Kritzman-Li financial turbulence)."""

import numpy as np
import pandas as pd
import pytest

from src.turbulence import financial_turbulence


def test_output_shape_and_index():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(60, 4)), columns=list("abcd"))
    t = financial_turbulence(df)
    assert len(t) == len(df)
    assert (t.index == df.index).all()
    assert t.name == "Financial turbulence"


def test_turbulence_is_non_negative():
    rng = np.random.default_rng(1)
    df = pd.DataFrame(rng.normal(size=(100, 3)), columns=list("xyz"))
    t = financial_turbulence(df)
    # Mahalanobis distance with a PSD covariance is non-negative
    assert (t >= -1e-9).all()


def test_outlier_row_has_high_turbulence():
    rng = np.random.default_rng(2)
    df = pd.DataFrame(rng.normal(scale=0.01, size=(120, 3)), columns=list("xyz"))
    df.iloc[50] = [0.5, -0.5, 0.5]  # an extreme, structure-breaking observation
    t = financial_turbulence(df)
    assert t.idxmax() == 50


def test_rejects_nan():
    df = pd.DataFrame({"a": [0.1, np.nan], "b": [0.2, 0.3]})
    with pytest.raises(ValueError):
        financial_turbulence(df)
