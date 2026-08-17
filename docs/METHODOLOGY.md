**English** | [Italiano](METHODOLOGY.it.md)

# Methodology

Technical summary of the **Equity Factor Timing** pipeline. For the full detail
see the report in [MANCUSO_Santo_PW.pdf](MANCUSO_Santo_PW.pdf) (in Italian); for
the step-by-step run see the notebook in `Code/`.

## Pipeline diagram

```
                  Part 1                      Part 2                     Part 3
        +-----------------------+   +----------------------+   +---------------------+
Data -> | Market regimes        |-> | Factor selection     |-> | Portfolio           |
        | (2-stage ML)          |   | (per regime)         |   | simulation          |
        +-----------------------+   +----------------------+   +---------------------+
          |                           |                          |
   K-means on drawdowns        Random Forest for            Weights = factor
   + classifier on             "normal" and                 probabilities; monthly
   macro/turbulence            "correction" regimes         rebalancing; metrics vs
                                                            benchmarks
```

## Part 1: market risk regimes

A **two-stage** approach:

1. **Unsupervised labelling.** The S&P 500's 3-month rolling drawdown is computed
   and aggregated to monthly frequency. A **K-means** model (number of clusters
   chosen via the *elbow method* and *silhouette score*) separates months into
   regimes. In the main configuration there are two regimes: **normal** and
   **correction**.
2. **Supervised classification.** A classifier learns to predict the month's
   regime from the features described below. **Random Forest**, **Gaussian Naive
   Bayes** and **SVC** are compared and combined into a **StackingClassifier**
   whose hyper-parameters are tuned with **HyperOpt** (temporal validation via
   `TimeSeriesSplit`). Class imbalance is handled with **SMOTE**.

### Regime classifier features

- **Financial turbulence** (Kritzman-Li index): Mahalanobis distance of the
  sector/treasury return vector from its historical mean. See
  `src/turbulence.py`.
- **Macroeconomic variables** (from `Data/Data_Macro.xlsx`), made stationary
  where needed and checked with the **Augmented Dickey-Fuller** test.
- Data is scaled with **Min-Max**, and feature selection uses **Random Forest
  importance** and **mutual information**.

## Part 2: factor selection

Six S&P 500 equity factors: **value, growth, momentum, low-volatility, quality,
small-cap**. For each month the "winning" factor is the one with the highest
return; the label is **one-hot**. **Two separate Random Forests** are trained,
one for the *normal* regime and one for the *correction* regime, each producing
the **probabilities** that a given factor will be next month's winner. The split
is temporal (train up to 2010, test from 2010 onwards).

## Part 3: portfolio simulation

The factor probabilities become the **weights** of a monthly-rebalanced
portfolio (the probabilities are lagged by one period to avoid look-ahead).
Several strategies are built and compared:

- **Factor-timing** portfolio driven by the probabilities (6 factors).
- **Equally-weighted** across the factors.
- **Risk-parity** (via `Riskfolio-Lib`).
- **S&P 500** buy-and-hold benchmark.

### Evaluation metrics

Computed on a monthly and daily basis by `src/metrics.py`:

| Metric | Meaning |
| --- | --- |
| `AnnRet` | Annualised return |
| `AnnVol` | Annualised volatility |
| `Sharpe` | Return per unit of risk |
| `MaxDD` | Maximum peak-to-trough loss |
| `Calmar` | Annualised return over maximum drawdown |

The numeric values for each strategy are produced by the Part 3 cells of the
notebook.

## Reproducibility notes

- The trained models are stored in `Code/*.pkl` and reloaded by the notebook;
  the heavy training cells are disabled by the `%%skip` magic.
- Retraining (by removing `%%skip`) may shift the results slightly because of
  seeds and library versions. See `requirements.txt` for the versions used.
