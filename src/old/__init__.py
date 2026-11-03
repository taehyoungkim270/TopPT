"""Simulation code for the time-delay embedding periodicity paper.

The implementation intentionally uses raw time-delay embeddings only.  No
phase folding is performed anywhere in this package.
"""

from .config import NoiseConfig, TDAConfig
from .periodicity_test import TDATestResult, run_tda_periodicity_test

__all__ = [
    "NoiseConfig",
    "TDAConfig",
    "TDATestResult",
    "run_tda_periodicity_test",
]
