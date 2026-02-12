import datetime
from typing import List, Tuple

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
import yfinance as yf


def get_spdr_sector_etfs() -> List[str]:
    """
    Lista de 11 Select Sector SPDR ETFs.
    """
    return ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "XLC", "XLRE"]


def fetch_history(ticker: str, trading_days: int = None, days: int = 365, months: int = None) -> pd.Series:
    end = datetime.date.today()
    if trading_days is not None:
        start = (pd.Timestamp(end) - BDay(trading_days)).date()
    elif months is not None:
        start = (pd.Timestamp(end) - pd.DateOffset(months=months)).date()
    else:
        start = end - datetime.timedelta(days=days)
    tk = yf.Ticker(ticker)
    df = tk.history(start=start.isoformat(), end=(end + datetime.timedelta(days=1)).isoformat(),
                    interval="1d", auto_adjust=True)
    if df.empty:
        return pd.Series(dtype=float)
    return df["Close"].dropna()


def twelve_month_return_from_series(prices: pd.Series) -> float:
    """
    Calcula el retorno simple entre el primer y último precio disponible
    en la serie (last / first - 1). Devuelve np.nan si no hay suficientes datos.
    """
    if prices.empty or len(prices) < 2:
        return np.nan
    first = prices.iloc[0]
    last = prices.iloc[-1]
    if first <= 0:
        return np.nan
    return float(last / first - 1.0)


def top_n_by_12m_return(tickers: List[str], n: int = 3, debug: bool = False) -> List[Tuple[str, float]]:
    """
    Descarga datos de los tickers (últimos 365 días hasta hoy) y devuelve
    los n mejores por retorno 12 meses.
    Resultado: lista de tuplas (ticker, retorno) ordenada de mayor a menor retorno.
    Si debug=True, imprime la fecha de inicio y fin de la serie y cantidad de filas.
    """
    results = []
    for t in tickers:
        try:
            s = fetch_history(t, days=365)
            if debug:
                if s.empty:
                    print(f"{t}: sin datos")
                else:
                    start = s.index[0].date()
                    end = s.index[-1].date()
                    print(f"{t}: start={start}, end={end}, rows={len(s)}")
            r = twelve_month_return_from_series(s)
            if np.isfinite(r):
                results.append((t, r))
        except Exception as e:
            if debug:
                print(f"{t}: error al descargar/procesar -> {e}")
            # omitir ticker si falla la descarga
            continue
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:n]


def top_n_by_months_return(tickers: List[str], n: int = 3, months: int = 11, debug: bool = False) -> List[Tuple[str, float]]:
    """
    Descarga datos de los tickers (últimos 'months' meses hasta hoy) y devuelve
    los n mejores por retorno en esa ventana.
    Resultado: lista de tuplas (ticker, retorno) ordenada de mayor a menor retorno.
    Si debug=True, imprime la fecha de inicio y fin de la serie y cantidad de filas.
    """
    results = []
    for t in tickers:
        try:
            s = fetch_history(t, months=months)
            if debug:
                if s.empty:
                    print(f"{t}: sin datos")
                else:
                    start = s.index[0].date()
                    end = s.index[-1].date()
                    print(f"{t}: start={start}, end={end}, rows={len(s)}")
            r = twelve_month_return_from_series(s)
            if np.isfinite(r):
                results.append((t, r))
        except Exception as e:
            if debug:
                print(f"{t}: error al descargar/procesar -> {e}")
            continue
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:n]


if __name__ == "__main__":
    etfs = get_spdr_sector_etfs()
    # activar debug=True para ver fechas de inicio/fin de cada serie
    top3 = top_n_by_months_return(etfs, n=3, months=11, debug=True)
    if not top3:
        print("No se obtuvieron datos válidos. Instalar yfinance y verificar conexión.")
    else:
        hoy = datetime.date.today().isoformat()
        print(f"Top 3 SPDR sector ETFs por retorno 11 meses (hasta {hoy}):")
        for tick, ret in top3:
            print(f"{tick}: {ret:.2%}")