"""Factor research and signal generation.

Provides a library of cross-sectional factors for asset ranking and
portfolio construction.  All factors implement the :class:`Factor` ABC.

Available Factors
-----------------
- :class:`MomentumFactor` — 12-month price momentum with FIP refinement
- :class:`SkewnessFactor` — Return skewness (long negative skew)
- :class:`LowVolatilityFactor` — Low volatility anomaly
- :class:`SectorMomentumFactor` — SPDR sector ETF rotation
"""

from quant.factors.base import Factor, rank_scores
from quant.factors.low_volatility import LowVolatilityFactor
from quant.factors.momentum import MomentumFactor
from quant.factors.registry import get_factor, list_factors, register_factor
from quant.factors.sector_rotation import SectorMomentumFactor
from quant.factors.skewness import SkewnessFactor

__all__ = [
    "Factor",
    "rank_scores",
    "MomentumFactor",
    "SkewnessFactor",
    "LowVolatilityFactor",
    "SectorMomentumFactor",
    "register_factor",
    "get_factor",
    "list_factors",
]
