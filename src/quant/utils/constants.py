"""Global constants used throughout the quant package.

Centralizes magic numbers that were previously scattered across
optimization/, performance/, strategies/, etc.
"""

# ── Trading calendar ─────────────────────────────────────────────────────────
TRADING_DAYS_PER_YEAR: int = 252
WEEKS_PER_YEAR: int = 52
MONTHS_PER_YEAR: int = 12
QUARTERS_PER_YEAR: int = 4

# ── Risk-free rate ───────────────────────────────────────────────────────────
# Default annualized risk-free rate (US T-bill proxy).
# Individual scripts may override this per market (e.g. Argentina).
DEFAULT_RISK_FREE_RATE: float = 0.05

# ── Default confidence levels ────────────────────────────────────────────────
DEFAULT_VAR_CONFIDENCE: float = 0.95
DEFAULT_CVAR_CONFIDENCE: float = 0.95

# ── Supported risk measures (Riskfolio-compatible codes) ─────────────────────
RISK_MEASURES: list[str] = [
    "MV",    # Mean Variance
    "MAD",   # Mean Absolute Deviation
    "MSV",   # Semi Variance
    "SLPM",  # Second Lower Partial Moment
    "CVaR",  # Conditional Value at Risk
    "VaR",   # Value at Risk
    "WR",    # Worst Realization
    "MDD",   # Maximum Drawdown
    "ADD",   # Average Drawdown
    "CDaR",  # Conditional Drawdown at Risk
    "EDaR",  # Entropic Drawdown at Risk
    "UCI",   # Ulcer Index
    "RG",    # Range
]

# ── Quarter-end dates (month, day) ──────────────────────────────────────────
QUARTER_END_DATES: list[tuple[int, int]] = [
    (3, 31),
    (6, 30),
    (9, 30),
    (12, 31),
]

# ── Transaction cost defaults ────────────────────────────────────────────────
DEFAULT_ROUND_TRIP_COST_BPS: float = 10.0  # 10 basis points
