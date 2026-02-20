"""Quantitative Value (QV) composite factor.

Combines franchise-power metrics (ROA, ROC, FCF/Assets, margin quality),
financial-strength flags (Piotroski-style F-Score), and an EBIT/EV
cheapness measure into a single ranking.

Methodology
-----------
1. **Franchise Power** — mean percentile of:
   - Eight-year geometric-mean ROA
   - Eight-year geometric-mean ROC
   - Cumulative 8yr FCF / current total assets
   - max(Margin Stability percentile, Margin Growth percentile)

2. **Financial Strength** — sum of 9 binary flags / 9:
   ROA>0, FCFTA>0, Accrual>0, Leverage improved, Liquidity improved,
   ROA change>0, FCFTA change>0, Margin change>0, Turnover change>0

3. **Composite score**::

       QUALITY = 0.5 * Franchise_Power + 0.5 * Financial_Strength
       QUANTITATIVE_VALUE = QUALITY + 2 * EBIT_EV_Percentile

Requires fundamental data (income, balance, cashflow, income_ttm) and
ticker info (enterprise_value, financial_currency) passed via ``**kwargs``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from quant.exceptions import DataError
from quant.factors.base import Factor, rank_scores
from quant.factors.registry import register_factor
from quant.utils.logging import get_logger

logger = get_logger(__name__)

# Minimum years of annual data required to compute multi-year metrics
_MIN_YEARS: int = 3
_MAX_YEARS: int = 8


def _safe_get(
    df: pd.DataFrame,
    ticker: str,
    field: str,
    col_idx: int = 0,
) -> float | None:
    """Safely extract a scalar from a MultiIndex statement DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Statement with MultiIndex ``(ticker, field)`` rows.
    ticker : str
        Ticker symbol.
    field : str
        Financial field name (e.g. ``'NetIncome'``).
    col_idx : int
        Index into non-NaN values (0 = most recent period).

    Returns
    -------
    float or None
        The value, or None if unavailable.
    """
    try:
        row = df.loc[(ticker, field)]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        # Drop NaN to get only actual data points, then index
        valid = row.dropna()
        if col_idx < len(valid):
            return float(valid.iloc[col_idx])
    except (KeyError, IndexError):
        pass
    return None


def _get_yearly_values(
    df: pd.DataFrame,
    ticker: str,
    field: str,
    max_years: int = _MAX_YEARS,
) -> list[float]:
    """Extract up to *max_years* annual non-NaN values for a field."""
    values: list[float] = []
    try:
        row = df.loc[(ticker, field)]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        # Drop NaN to get only actual data points
        valid = row.dropna()
        for i in range(min(max_years, len(valid))):
            values.append(float(valid.iloc[i]))
    except (KeyError, IndexError):
        pass
    return values


def _geometric_mean(values: list[float]) -> float | None:
    """Compute geometric mean of (1 + r_i) - 1 from a list of ratios."""
    clean = [v for v in values if np.isfinite(v)]
    if len(clean) < _MIN_YEARS:
        return None
    factors = [1 + v for v in clean]
    product = np.prod(factors)
    if product <= 0:
        return None
    result = product ** (1 / len(clean)) - 1
    if isinstance(result, complex):
        return None
    return float(result)


def _percentile_scores(values: pd.Series) -> pd.Series:
    """Compute percentile-of-score for each value (0-1 scale)."""
    clean = values.dropna()
    if clean.empty:
        return pd.Series(np.nan, index=values.index)
    arr = clean.values
    pcts = pd.Series(np.nan, index=values.index)
    for idx in clean.index:
        pcts[idx] = stats.percentileofscore(arr, clean[idx]) / 100.0
    return pcts


def _compute_ticker_metrics(
    ticker: str,
    income: pd.DataFrame,
    balance: pd.DataFrame,
    cashflow: pd.DataFrame,
    income_ttm: pd.DataFrame,
    ticker_info: pd.Series | None,
) -> dict[str, Any]:
    """Compute all QV metrics for a single ticker.

    Returns a dict with raw metrics and binary F-Score flags.
    """
    result: dict[str, Any] = {"ticker": ticker}

    # Determine how many years of data we have
    has_income = ticker in income.index.get_level_values(0)
    has_balance = ticker in balance.index.get_level_values(0)
    has_cashflow = ticker in cashflow.index.get_level_values(0)

    if not (has_income and has_balance):
        return result

    # Number of periods with actual data (non-NaN)
    def _count_valid_periods(df: pd.DataFrame, tkr: str, field: str) -> int:
        try:
            row = df.loc[(tkr, field)]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            return int(row.dropna().count())
        except (KeyError, IndexError):
            return 0

    n_income = _count_valid_periods(income, ticker, "NetIncome")
    n_balance = _count_valid_periods(balance, ticker, "TotalAssets")
    years = min(_MAX_YEARS, max(n_income, 1), max(n_balance, 1))
    result["years_data"] = years

    # ── Eight-Year ROA (geometric mean of NetIncome / TotalAssets) ──
    net_incomes = _get_yearly_values(income, ticker, "NetIncome", years)
    total_assets = _get_yearly_values(balance, ticker, "TotalAssets", years)
    roa_values = []
    for ni, ta in zip(net_incomes, total_assets):
        if np.isfinite(ni) and np.isfinite(ta) and ta != 0:
            roa_values.append(ni / ta)
    result["eight_year_roa"] = _geometric_mean(roa_values)

    # ── Eight-Year ROC (OperatingIncome / Capital) ──
    op_incomes = _get_yearly_values(income, ticker, "OperatingIncome", years)
    current_assets_vals = _get_yearly_values(balance, ticker, "CurrentAssets", years)
    current_liab_vals = _get_yearly_values(balance, ticker, "CurrentLiabilities", years)
    net_ppe_vals = _get_yearly_values(balance, ticker, "NetPPE", years)
    roc_values = []
    for oi, ca, cl, ppe in zip(op_incomes, current_assets_vals, current_liab_vals, net_ppe_vals):
        if all(np.isfinite(v) for v in [oi, ca, cl, ppe]):
            capital = (ca - cl) + ppe
            if capital > 0:
                roc_values.append(oi / capital)
    result["eight_year_roc"] = _geometric_mean(roc_values)

    # ── FCF Sum / Total Assets ──
    if has_cashflow:
        fcf_vals = _get_yearly_values(cashflow, ticker, "FreeCashFlow", years)
        fcf_clean = [v for v in fcf_vals if np.isfinite(v)]
        fcf_sum = sum(fcf_clean) if fcf_clean else 0.0
        current_ta = _safe_get(balance, ticker, "TotalAssets", 0)
        if fcf_sum != 0 and current_ta is not None and current_ta != 0:
            result["fcf_sum_to_assets"] = fcf_sum / current_ta
        else:
            result["fcf_sum_to_assets"] = None
    else:
        result["fcf_sum_to_assets"] = None

    # ── Gross Margin Growth & Stability ──
    gross_profits = _get_yearly_values(income, ticker, "GrossProfit", years)
    revenues = _get_yearly_values(income, ticker, "TotalRevenue", years)
    gross_margins: list[float] = []
    for gp, rev in zip(gross_profits, revenues):
        if np.isfinite(gp) and np.isfinite(rev) and rev != 0:
            gross_margins.append(gp / rev)

    if len(gross_margins) >= _MIN_YEARS:
        avg_margin = np.mean(gross_margins)
        std_margin = np.std(gross_margins, ddof=1)
        result["margin_stability"] = avg_margin / std_margin if std_margin > 0 else None
    else:
        result["margin_stability"] = None

    if len(gross_margins) > 1 and gross_margins[0] > 0 and gross_margins[-1] > 0:
        ratio = gross_margins[0] / gross_margins[-1]
        if ratio > 0:
            growth = ratio ** (1 / (len(gross_margins) - 1)) - 1
            result["margin_growth"] = float(growth) if not isinstance(growth, complex) else None
        else:
            result["margin_growth"] = None
    else:
        result["margin_growth"] = None

    # ── EBIT/EV (TTM OperatingIncome / EnterpriseValue) ──
    has_ttm = ticker in income_ttm.index.get_level_values(0) if not income_ttm.empty else False
    operating_income_ttm = _safe_get(income_ttm, ticker, "OperatingIncome") if has_ttm else None
    ev = float(ticker_info["enterprise_value"]) if (
        ticker_info is not None and pd.notna(ticker_info.get("enterprise_value"))
    ) else None

    if operating_income_ttm is not None and ev is not None and ev > 0:
        result["ebit_ev"] = operating_income_ttm / ev
    else:
        result["ebit_ev"] = None

    # ── F-Score Flags ──
    # 1. ROA > 0
    current_roa = None
    ni_0 = _safe_get(income, ticker, "NetIncome", 0)
    ta_0 = _safe_get(balance, ticker, "TotalAssets", 0)
    if ni_0 is not None and ta_0 is not None and ta_0 != 0:
        current_roa = ni_0 / ta_0
    result["fs_roa"] = 1 if (current_roa is not None and current_roa > 0) else 0

    # 2. FCFTA > 0
    current_fcfta = None
    if has_cashflow:
        fcf_0 = _safe_get(cashflow, ticker, "FreeCashFlow", 0)
        if fcf_0 is not None and ta_0 is not None and ta_0 != 0:
            current_fcfta = fcf_0 / ta_0
    result["fs_fcfta"] = 1 if (current_fcfta is not None and current_fcfta > 0) else 0

    # 3. Accrual = FCFTA - ROA > 0
    if current_fcfta is not None and current_roa is not None:
        result["fs_accrual"] = 1 if (current_fcfta - current_roa) > 0 else 0
    else:
        result["fs_accrual"] = 0

    # 4. Leverage improved (LT debt / TA decreased)
    if years >= 2:
        ltd_0 = _safe_get(balance, ticker, "LongTermDebt", 0)
        ltd_1 = _safe_get(balance, ticker, "LongTermDebt", 1)
        ta_1 = _safe_get(balance, ticker, "TotalAssets", 1)
        if all(v is not None for v in [ltd_0, ta_0, ltd_1, ta_1]) and ta_0 > 0 and ta_1 > 0:
            result["fs_lever"] = 1 if (ltd_1 / ta_1 - ltd_0 / ta_0) > 0 else 0
        else:
            result["fs_lever"] = 0
    else:
        result["fs_lever"] = 0

    # 5. Liquidity improved (current ratio increased)
    if years >= 2:
        ca_0 = _safe_get(balance, ticker, "CurrentAssets", 0)
        cl_0 = _safe_get(balance, ticker, "CurrentLiabilities", 0)
        ca_1 = _safe_get(balance, ticker, "CurrentAssets", 1)
        cl_1 = _safe_get(balance, ticker, "CurrentLiabilities", 1)
        if all(v is not None for v in [ca_0, cl_0, ca_1, cl_1]) and cl_0 > 0 and cl_1 > 0:
            result["fs_liquid"] = 1 if (ca_0 / cl_0 - ca_1 / cl_1) > 0 else 0
        else:
            result["fs_liquid"] = 0
    else:
        result["fs_liquid"] = 0

    # 6. ROA change > 0
    if years >= 2:
        ni_1 = _safe_get(income, ticker, "NetIncome", 1)
        if all(v is not None for v in [ni_0, ta_0, ni_1, ta_1]) and ta_0 > 0 and ta_1 > 0:
            result["fs_roa_change"] = 1 if (ni_0 / ta_0 - ni_1 / ta_1) > 0 else 0
        else:
            result["fs_roa_change"] = 0
    else:
        result["fs_roa_change"] = 0

    # 7. FCFTA change > 0
    if years >= 2 and has_cashflow:
        fcf_1 = _safe_get(cashflow, ticker, "FreeCashFlow", 1)
        if all(v is not None for v in [fcf_0, ta_0, fcf_1, ta_1]) and ta_0 > 0 and ta_1 > 0:
            result["fs_fcfta_change"] = 1 if (fcf_0 / ta_0 - fcf_1 / ta_1) > 0 else 0
        else:
            result["fs_fcfta_change"] = 0
    else:
        result["fs_fcfta_change"] = 0

    # 8. Margin change > 0 (gross margin YoY)
    if len(gross_margins) >= 2:
        result["fs_margin"] = 1 if (gross_margins[0] - gross_margins[1]) > 0 else 0
    else:
        result["fs_margin"] = 0

    # 9. Turnover change > 0 (asset turnover = revenue / total assets)
    if years >= 2:
        rev_0 = _safe_get(income, ticker, "TotalRevenue", 0)
        rev_1 = _safe_get(income, ticker, "TotalRevenue", 1)
        if all(v is not None for v in [rev_0, ta_0, rev_1, ta_1]) and ta_0 > 0 and ta_1 > 0:
            result["fs_turn"] = 1 if (rev_0 / ta_0 - rev_1 / ta_1) > 0 else 0
        else:
            result["fs_turn"] = 0
    else:
        result["fs_turn"] = 0

    return result


@register_factor("quantitative_value")
class QuantitativeValueFactor(Factor):
    """Quantitative Value composite factor.

    Combines franchise-power metrics, financial-strength flags, and
    EBIT/EV cheapness into a single ranking score.

    Parameters
    ----------
    min_market_cap : float
        Minimum market capitalisation in USD (default $1.4B).
    usd_only : bool
        If True (default), restrict to tickers reporting in USD.
    ebit_ev_top_pct : float
        Keep only the top X% by EBIT/EV before final ranking (default 0.20).
    """

    requires_fundamentals: bool = True

    def __init__(
        self,
        min_market_cap: float = 1.4e9,
        usd_only: bool = True,
        ebit_ev_top_pct: float = 0.20,
    ) -> None:
        super().__init__(name="Quantitative_Value", lookback=0)
        self.min_market_cap = min_market_cap
        self.usd_only = usd_only
        self.ebit_ev_top_pct = ebit_ev_top_pct

    def validate_data(self, prices: pd.DataFrame, **kwargs: Any) -> None:
        """Validate that fundamental data is provided via kwargs."""
        fundamentals = kwargs.get("fundamentals")
        if fundamentals is None:
            raise DataError(
                "QuantitativeValueFactor requires 'fundamentals' kwarg — "
                "a dict with keys 'income', 'balance', 'cashflow', 'income_ttm'."
            )
        required_keys = {"income", "balance"}
        missing = required_keys - set(fundamentals.keys())
        if missing:
            raise DataError(f"Missing fundamental statements: {missing}")

        ticker_info = kwargs.get("ticker_info")
        if ticker_info is None:
            raise DataError(
                "QuantitativeValueFactor requires 'ticker_info' kwarg — "
                "a DataFrame with enterprise_value and financial_currency columns."
            )

    def compute(self, prices: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Compute Quantitative Value scores.

        Parameters
        ----------
        prices : pd.DataFrame
            Daily adjusted close prices (DatetimeIndex x tickers).
            Used only to determine the universe of tickers to score.
        **kwargs
            Must include:

            - ``fundamentals`` : dict[str, pd.DataFrame] — output of
              ``FundamentalStore.load()``
            - ``ticker_info`` : pd.DataFrame — ticker metadata with columns
              ``enterprise_value``, ``financial_currency``, ``market_cap``

        Returns
        -------
        pd.DataFrame
            Columns: ``['ticker', 'qv_score', 'quality', 'franchise_power',
            'financial_strength', 'ebit_ev', 'score', 'rank']``.
        """
        self.validate_data(prices, **kwargs)

        fundamentals: dict[str, pd.DataFrame] = kwargs["fundamentals"]
        ticker_info: pd.DataFrame = kwargs["ticker_info"]

        income = fundamentals.get("income", pd.DataFrame())
        balance = fundamentals.get("balance", pd.DataFrame())
        cashflow = fundamentals.get("cashflow", pd.DataFrame())
        income_ttm = fundamentals.get("income_ttm", pd.DataFrame())

        # Determine which tickers have fundamental data
        fund_tickers = set()
        if not income.empty:
            fund_tickers.update(income.index.get_level_values(0).unique())
        # Intersect with price universe
        universe = [t for t in prices.columns if t in fund_tickers]

        if not universe:
            raise DataError("No tickers with both price and fundamental data found.")

        # ── Pre-filter by currency and market cap ──
        filtered_universe: list[str] = []
        for ticker in universe:
            if ticker not in ticker_info.index:
                continue
            info_row = ticker_info.loc[ticker]

            if self.usd_only:
                currency = str(info_row.get("financial_currency", ""))
                if currency.upper() != "USD":
                    continue

            cap = info_row.get("market_cap", np.nan)
            if pd.notna(cap) and cap < self.min_market_cap:
                continue

            filtered_universe.append(ticker)

        if not filtered_universe:
            raise DataError(
                "No tickers passed pre-filters "
                f"(USD only={self.usd_only}, min cap=${self.min_market_cap/1e9:.1f}B)."
            )

        logger.info("Computing QV metrics for %d tickers...", len(filtered_universe))

        # ── Compute per-ticker metrics ──
        records: list[dict[str, Any]] = []
        for ticker in filtered_universe:
            info_row = ticker_info.loc[ticker] if ticker in ticker_info.index else None
            metrics = _compute_ticker_metrics(
                ticker, income, balance, cashflow, income_ttm, info_row,
            )
            records.append(metrics)

        df = pd.DataFrame(records)
        if df.empty or "ticker" not in df.columns:
            raise DataError("No valid metrics computed for any ticker.")

        # ── Percentile rankings ──
        pct_cols = {
            "eight_year_roa": "roa_pct",
            "eight_year_roc": "roc_pct",
            "fcf_sum_to_assets": "fcf_pct",
            "margin_stability": "margin_stability_pct",
            "margin_growth": "margin_growth_pct",
            "ebit_ev": "ebit_ev_pct",
        }
        for raw_col, pct_col in pct_cols.items():
            if raw_col in df.columns:
                df[pct_col] = _percentile_scores(df[raw_col])
            else:
                df[pct_col] = np.nan

        # Max margin metric
        df["max_margin_pct"] = df[["margin_stability_pct", "margin_growth_pct"]].max(axis=1)

        # ── Franchise Power = mean(ROA_pct, ROC_pct, FCF_pct, Max_Margin_pct) ──
        fp_cols = ["roa_pct", "roc_pct", "fcf_pct", "max_margin_pct"]
        df["franchise_power"] = df[fp_cols].mean(axis=1, skipna=True)

        # ── Financial Strength = sum of 9 flags / 9 ──
        fs_cols = [
            "fs_roa", "fs_fcfta", "fs_accrual", "fs_lever", "fs_liquid",
            "fs_roa_change", "fs_fcfta_change", "fs_margin", "fs_turn",
        ]
        existing_fs = [c for c in fs_cols if c in df.columns]
        df["financial_strength"] = df[existing_fs].sum(axis=1) / 9.0

        # ── QUALITY and QV Score ──
        df["quality"] = 0.5 * df["franchise_power"] + 0.5 * df["financial_strength"]
        df["qv_score"] = df["quality"] + 2.0 * df["ebit_ev_pct"]

        # ── Filter to top EBIT/EV tickers ──
        ebit_valid = df.dropna(subset=["ebit_ev"])
        if not ebit_valid.empty:
            threshold = ebit_valid["ebit_ev"].quantile(1.0 - self.ebit_ev_top_pct)
            df = df[df["ebit_ev"].isna() | (df["ebit_ev"] >= threshold)]

        # Drop rows without a computable QV score
        df = df.dropna(subset=["qv_score"])

        if df.empty:
            raise DataError("No tickers with computable Quantitative Value score.")

        # ── Final output columns ──
        df["score"] = df["qv_score"]
        df["rank"] = rank_scores(df["score"]).values

        output_cols = [
            "ticker", "qv_score", "quality", "franchise_power",
            "financial_strength", "ebit_ev", "score", "rank",
        ]
        return (
            df[output_cols]
            .sort_values("rank", ascending=False)
            .reset_index(drop=True)
        )
