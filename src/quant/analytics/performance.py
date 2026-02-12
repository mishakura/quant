"""Consolidated performance analytics — the single authoritative source for
all return, risk, drawdown, and benchmark metrics.

Replaces the duplicated calculations scattered across ``optimization/h.py``,
``optimization/nco.py``, ``optimization/mincorr.py``,
``performance/performance.py``, ``strategies/trend_following/stats.py``,
and ``strategies/hedging/dynamic_hedge/dhedging.py``.

All functions operate on **arithmetic (simple) daily returns** unless
otherwise noted.  Annualization defaults to 252 trading days.
"""

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from quant.utils.constants import DEFAULT_RISK_FREE_RATE, TRADING_DAYS_PER_YEAR


# ═══════════════════════════════════════════════════════════════════════════════
# Return metrics
# ═══════════════════════════════════════════════════════════════════════════════

def total_return(returns: pd.Series) -> float:
    """Cumulative total return: ``(1 + r_1)(1 + r_2)...(1 + r_T) - 1``.

    Parameters
    ----------
    returns : pd.Series
        Simple (arithmetic) period returns.

    Returns
    -------
    float
    """
    return float((1 + returns).prod() - 1)


def annualized_return(
    returns: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Compound annual growth rate (CAGR).

    Parameters
    ----------
    returns : pd.Series
        Simple period returns.
    periods_per_year : int
        Number of periods per year for annualization.

    Returns
    -------
    float
    """
    n = len(returns)
    if n == 0:
        return np.nan
    total = (1 + returns).prod()
    return float(total ** (periods_per_year / n) - 1)


def cagr(
    returns: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Alias for :func:`annualized_return`."""
    return annualized_return(returns, periods_per_year)


# ═══════════════════════════════════════════════════════════════════════════════
# Volatility metrics
# ═══════════════════════════════════════════════════════════════════════════════

def annualized_volatility(
    returns: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualized standard deviation of returns.

    Parameters
    ----------
    returns : pd.Series
        Simple period returns.
    periods_per_year : int
        Annualization factor.

    Returns
    -------
    float
    """
    return float(returns.std() * np.sqrt(periods_per_year))


def downside_deviation(
    returns: pd.Series,
    threshold: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualized downside deviation (semi-deviation below threshold).

    Parameters
    ----------
    returns : pd.Series
        Simple period returns.
    threshold : float
        Minimum acceptable return per period (default: 0).
    periods_per_year : int
        Annualization factor.

    Returns
    -------
    float
    """
    diff = returns - threshold
    downside = diff[diff < 0]
    if len(downside) == 0:
        return 0.0
    return float(downside.std() * np.sqrt(periods_per_year))


# ═══════════════════════════════════════════════════════════════════════════════
# Risk-adjusted return ratios
# ═══════════════════════════════════════════════════════════════════════════════

def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualized Sharpe ratio.

    Computed as ``(mean_excess * periods) / (std * sqrt(periods))``,
    equivalent to ``annualized_excess / annualized_vol``.

    Parameters
    ----------
    returns : pd.Series
        Simple period returns.
    risk_free_rate : float
        Annualized risk-free rate.
    periods_per_year : int
        Annualization factor.

    Returns
    -------
    float
    """
    rf_per_period = risk_free_rate / periods_per_year
    excess = returns - rf_per_period
    if returns.std() == 0:
        return np.nan
    return float(excess.mean() / returns.std() * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualized Sortino ratio.

    Uses downside deviation instead of total volatility.

    Parameters
    ----------
    returns : pd.Series
        Simple period returns.
    risk_free_rate : float
        Annualized risk-free rate.
    periods_per_year : int
        Annualization factor.

    Returns
    -------
    float
    """
    rf_per_period = risk_free_rate / periods_per_year
    excess = returns - rf_per_period
    dd = downside_deviation(returns, threshold=rf_per_period, periods_per_year=periods_per_year)
    if dd == 0:
        return np.nan
    ann_excess = excess.mean() * periods_per_year
    return float(ann_excess / dd)


def calmar_ratio(
    returns: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Calmar ratio: annualized return / |max drawdown|.

    Parameters
    ----------
    returns : pd.Series
        Simple period returns.
    periods_per_year : int
        Annualization factor.

    Returns
    -------
    float
    """
    mdd = max_drawdown(returns)
    if mdd == 0:
        return np.nan
    ann_ret = annualized_return(returns, periods_per_year)
    return float(ann_ret / abs(mdd))


def omega_ratio(
    returns: pd.Series,
    threshold: float = 0.0,
) -> float:
    """Omega ratio: probability-weighted gains over losses above threshold.

    Parameters
    ----------
    returns : pd.Series
        Simple period returns.
    threshold : float
        Return threshold (per period).

    Returns
    -------
    float
    """
    excess = returns - threshold
    gains = excess[excess > 0].sum()
    losses = abs(excess[excess <= 0].sum())
    if losses == 0:
        return np.nan
    return float(gains / losses)


def information_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Information ratio: annualized active return / tracking error.

    Parameters
    ----------
    returns : pd.Series
        Portfolio returns.
    benchmark_returns : pd.Series
        Benchmark returns (aligned to same dates).
    periods_per_year : int
        Annualization factor.

    Returns
    -------
    float
    """
    te = tracking_error(returns, benchmark_returns, periods_per_year)
    if te == 0:
        return np.nan
    active = returns - benchmark_returns
    ann_active = active.mean() * periods_per_year
    return float(ann_active / te)


# ═══════════════════════════════════════════════════════════════════════════════
# Drawdown metrics
# ═══════════════════════════════════════════════════════════════════════════════

def drawdown_series(returns: pd.Series) -> pd.Series:
    """Compute the drawdown time series from returns.

    Parameters
    ----------
    returns : pd.Series
        Simple period returns.

    Returns
    -------
    pd.Series
        Drawdown values (negative or zero) at each point in time.
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    return cumulative / running_max - 1


def max_drawdown(returns: pd.Series) -> float:
    """Maximum drawdown (most negative peak-to-trough decline).

    Parameters
    ----------
    returns : pd.Series
        Simple period returns.

    Returns
    -------
    float
        Negative value representing the worst drawdown.
    """
    dd = drawdown_series(returns)
    if dd.empty:
        return 0.0
    return float(dd.min())


def max_drawdown_duration(returns: pd.Series) -> int:
    """Duration of the longest drawdown (in periods).

    Parameters
    ----------
    returns : pd.Series
        Simple period returns.

    Returns
    -------
    int
        Number of periods in the longest drawdown episode.
    """
    dd = drawdown_series(returns)
    in_drawdown = dd < 0
    if not in_drawdown.any():
        return 0
    groups = (in_drawdown != in_drawdown.shift()).cumsum()
    dd_lengths = in_drawdown[in_drawdown].groupby(groups[in_drawdown]).size()
    return int(dd_lengths.max()) if not dd_lengths.empty else 0


def average_drawdown(returns: pd.Series) -> float:
    """Average drawdown across all drawdown episodes.

    Parameters
    ----------
    returns : pd.Series
        Simple period returns.

    Returns
    -------
    float
        Mean of the per-episode average drawdown values.
    """
    dd = drawdown_series(returns)
    in_drawdown = dd < 0
    if not in_drawdown.any():
        return 0.0
    groups = (in_drawdown != in_drawdown.shift()).cumsum()
    episode_means = dd[in_drawdown].groupby(groups[in_drawdown]).mean()
    return float(episode_means.mean())


# ═══════════════════════════════════════════════════════════════════════════════
# Tail risk metrics
# ═══════════════════════════════════════════════════════════════════════════════

def value_at_risk(
    returns: pd.Series,
    confidence: float = 0.95,
    method: str = "historical",
) -> float:
    """Value at Risk (VaR).

    Parameters
    ----------
    returns : pd.Series
        Simple period returns.
    confidence : float
        Confidence level (e.g. 0.95 for 95%).
    method : str
        ``'historical'`` (quantile-based) or ``'parametric'`` (Gaussian).

    Returns
    -------
    float
        The VaR threshold (a negative number for losses).
    """
    alpha = 1 - confidence
    if method == "historical":
        return float(returns.quantile(alpha))
    elif method == "parametric":
        z = sp_stats.norm.ppf(alpha)
        return float(returns.mean() + z * returns.std())
    else:
        raise ValueError(f"Unknown VaR method '{method}'. Use 'historical' or 'parametric'.")


def conditional_var(
    returns: pd.Series,
    confidence: float = 0.95,
) -> float:
    """Conditional Value at Risk (CVaR / Expected Shortfall).

    Mean of returns that fall at or below the VaR threshold.

    Parameters
    ----------
    returns : pd.Series
        Simple period returns.
    confidence : float
        Confidence level.

    Returns
    -------
    float
    """
    var = value_at_risk(returns, confidence, method="historical")
    tail = returns[returns <= var]
    if tail.empty:
        return var
    return float(tail.mean())


def tail_ratio(returns: pd.Series) -> float:
    """Tail ratio: 95th percentile / |5th percentile|.

    Measures the symmetry of the return distribution tails.
    Values > 1 indicate a fatter right tail (positive skew in tails).

    Parameters
    ----------
    returns : pd.Series
        Simple period returns.

    Returns
    -------
    float
    """
    q95 = returns.quantile(0.95)
    q05 = returns.quantile(0.05)
    if q05 == 0:
        return np.nan
    return float(q95 / abs(q05))


# ═══════════════════════════════════════════════════════════════════════════════
# Distribution metrics
# ═══════════════════════════════════════════════════════════════════════════════

def return_skewness(returns: pd.Series) -> float:
    """Skewness of the return distribution.

    Parameters
    ----------
    returns : pd.Series
        Simple period returns.

    Returns
    -------
    float
        Positive = right-skewed, negative = left-skewed.
    """
    return float(sp_stats.skew(returns, nan_policy="omit"))


def return_kurtosis(returns: pd.Series) -> float:
    """Excess kurtosis of the return distribution.

    Parameters
    ----------
    returns : pd.Series
        Simple period returns.

    Returns
    -------
    float
        Excess kurtosis (0 for normal distribution).
    """
    return float(sp_stats.kurtosis(returns, nan_policy="omit"))


def win_rate(returns: pd.Series) -> float:
    """Fraction of periods with positive returns.

    Parameters
    ----------
    returns : pd.Series
        Simple period returns.

    Returns
    -------
    float
        Value between 0 and 1.
    """
    if len(returns) == 0:
        return np.nan
    return float((returns > 0).mean())


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark-relative metrics
# ═══════════════════════════════════════════════════════════════════════════════

def beta(
    returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """CAPM beta: ``Cov(r, r_m) / Var(r_m)``.

    Parameters
    ----------
    returns : pd.Series
        Portfolio returns.
    benchmark_returns : pd.Series
        Market/benchmark returns (same index).

    Returns
    -------
    float
    """
    aligned = pd.concat([returns, benchmark_returns], axis=1, join="inner").dropna()
    if len(aligned) < 2:
        return np.nan
    cov = aligned.iloc[:, 0].cov(aligned.iloc[:, 1])
    var = aligned.iloc[:, 1].var()
    if var == 0:
        return np.nan
    return float(cov / var)


def jensens_alpha(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Jensen's alpha: excess return not explained by market exposure.

    ``alpha = R_p - [R_f + beta * (R_m - R_f)]`` (annualized).

    Parameters
    ----------
    returns : pd.Series
        Portfolio returns.
    benchmark_returns : pd.Series
        Market/benchmark returns.
    risk_free_rate : float
        Annualized risk-free rate.
    periods_per_year : int
        Annualization factor.

    Returns
    -------
    float
    """
    aligned = pd.concat([returns, benchmark_returns], axis=1, join="inner").dropna()
    if len(aligned) < 2:
        return np.nan
    port_ann = aligned.iloc[:, 0].mean() * periods_per_year
    bench_ann = aligned.iloc[:, 1].mean() * periods_per_year
    b = beta(returns, benchmark_returns)
    if np.isnan(b):
        return np.nan
    return float(port_ann - risk_free_rate - b * (bench_ann - risk_free_rate))


def tracking_error(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualized tracking error: std dev of active returns.

    Parameters
    ----------
    returns : pd.Series
        Portfolio returns.
    benchmark_returns : pd.Series
        Benchmark returns.
    periods_per_year : int
        Annualization factor.

    Returns
    -------
    float
    """
    aligned = pd.concat([returns, benchmark_returns], axis=1, join="inner").dropna()
    if len(aligned) < 2:
        return np.nan
    active = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    return float(active.std() * np.sqrt(periods_per_year))


# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════

def performance_summary(
    returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> dict[str, float]:
    """Comprehensive performance report in a single call.

    Parameters
    ----------
    returns : pd.Series
        Portfolio simple period returns.
    benchmark_returns : pd.Series or None
        Benchmark returns for relative metrics. If None, benchmark-relative
        metrics are set to NaN.
    risk_free_rate : float
        Annualized risk-free rate.
    periods_per_year : int
        Annualization factor.

    Returns
    -------
    dict[str, float]
        Dictionary of all key performance metrics.
    """
    rf_per_period = risk_free_rate / periods_per_year

    summary: dict[str, float] = {
        "Total Return": total_return(returns),
        "Annualized Return": annualized_return(returns, periods_per_year),
        "Annualized Volatility": annualized_volatility(returns, periods_per_year),
        "Downside Deviation": downside_deviation(returns, rf_per_period, periods_per_year),
        "Sharpe Ratio": sharpe_ratio(returns, risk_free_rate, periods_per_year),
        "Sortino Ratio": sortino_ratio(returns, risk_free_rate, periods_per_year),
        "Calmar Ratio": calmar_ratio(returns, periods_per_year),
        "Omega Ratio": omega_ratio(returns, rf_per_period),
        "Max Drawdown": max_drawdown(returns),
        "Max Drawdown Duration": float(max_drawdown_duration(returns)),
        "Average Drawdown": average_drawdown(returns),
        "VaR 95%": value_at_risk(returns, 0.95),
        "CVaR 95%": conditional_var(returns, 0.95),
        "Tail Ratio": tail_ratio(returns),
        "Skewness": return_skewness(returns),
        "Kurtosis": return_kurtosis(returns),
        "Win Rate": win_rate(returns),
    }

    if benchmark_returns is not None:
        summary["Beta"] = beta(returns, benchmark_returns)
        summary["Jensen's Alpha"] = jensens_alpha(
            returns, benchmark_returns, risk_free_rate, periods_per_year
        )
        summary["Tracking Error"] = tracking_error(
            returns, benchmark_returns, periods_per_year
        )
        summary["Information Ratio"] = information_ratio(
            returns, benchmark_returns, periods_per_year
        )
    else:
        summary["Beta"] = np.nan
        summary["Jensen's Alpha"] = np.nan
        summary["Tracking Error"] = np.nan
        summary["Information Ratio"] = np.nan

    return summary
