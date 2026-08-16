"""
Reusable library for the Equity Factor Timing project.

Modules
-------
paths          : project paths resolved independently of the working directory.
metrics        : performance metrics computed from a return series (used as `pm`).
equity_metrics : the same metrics computed from an equity curve.
turbulence     : Kritzman-Li financial turbulence index.
"""

from . import paths  # noqa: F401
