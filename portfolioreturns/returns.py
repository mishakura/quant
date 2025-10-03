from pathlib import Path
from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
import datetime
import os
import plotly.graph_objects as go

def load_weights(weights_path):
    xls = pd.read_excel(weights_path, sheet_name=None)
    portfolios = {}
    for sheet, df in xls.items():
        df = df.copy()
        # Normalizar nombres de columnas esperadas
        df.columns = [c.strip() for c in df.columns]
        if 'Ticker' not in df.columns or 'Weight' not in df.columns:
            raise ValueError(f'La hoja "{sheet}" debe tener columnas "Ticker" y "Weight".')
        df['Ticker'] = df['Ticker'].astype(str).str.strip().str.upper()
        df['Weight'] = pd.to_numeric(df['Weight'], errors='coerce').fillna(0.0)
        # Si todos los pesos son cero, asignar pesos iguales entre tickers no vacíos
        sum_w = df['Weight'].sum()
        valid_tickers = df['Ticker'].loc[df['Ticker'].str.strip() != '']
        if sum_w == 0:
            n = len(valid_tickers)
            if n == 0:
                raise ValueError(f'La hoja "{sheet}" no contiene tickers válidos.')
            df.loc[valid_tickers.index, 'Weight'] = 1.0 / n
        else:
            # Normalizar pesos a suma 1 (si no suman exactamente 1)
            df['Weight'] = df['Weight'] / sum_w
        portfolios[sheet] = df
    return portfolios

def load_indices(indices_path):
    # Lee todas las hojas; cada hoja debe tener Fecha y Precio (o primeras 2 columnas)
    xls = pd.read_excel(indices_path, sheet_name=None, parse_dates=[0])
    series = {}
    for sheet, df in xls.items():
        df = df.copy()
        # si hay columnas explícitas Fecha/Precio úsalas, si no toma las dos primeras
        if 'Fecha' in df.columns and 'Precio' in df.columns:
            s = df[['Fecha', 'Precio']].dropna()
            s = s.set_index('Fecha').squeeze()
        else:
            # fallback: primera columna fechas, segunda columna precios
            s = df.iloc[:, :2].dropna()
            s.columns = ['Fecha', 'Precio']
            s = s.set_index('Fecha').squeeze()
        s.index = pd.to_datetime(s.index)
        s.name = sheet.upper()
        series[s.name] = s.sort_index()
    return series  # dict of Series

def download_market_prices(tickers, start, end=None):
    # tickers: list of tickers (already uppercased) excluding any special-case tickers handled separately
    if not tickers:
        return pd.DataFrame()
    # yfinance can accept list; auto_adjust True
    if end is not None:
        df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    else:
        df = yf.download(tickers, start=start, auto_adjust=True, progress=False)
    if df.empty:
        return pd.DataFrame()
    if 'Close' in df.columns:
        prices = df['Close']
    else:
        # if single ticker, yfinance may return a single-level DF/Series
        prices = df
    # Ensure DataFrame
    prices = prices.copy()
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(prices.name)
    # Uppercase column names
    prices.columns = [c.upper() for c in prices.columns]
    prices.index = pd.to_datetime(prices.index)
    return prices

def compute_stats_from_returns(returns, rf=0.0):
    # returns: pd.Series of periodic returns (monthly or daily)
    if returns.dropna().empty:
        return {}
    # infer periodicity: si la mediana de días entre observaciones >= 20 => mensual (aprox 30d)
    try:
        med_days = returns.index.to_series().diff().dt.days.dropna().median()
    except Exception:
        med_days = None
    # usar 12 para mensual, 252 para diario; umbral ajustado a 20 días
    periods_per_year = 12 if (med_days is not None and med_days >= 20) else 252

    n = returns.count()
    days = (returns.index[-1] - returns.index[0]).days
    cumulative = (1 + returns).cumprod()
    total_return = cumulative.iloc[-1] - 1
    years = days / 365.25 if days > 0 else n / periods_per_year
    if years <= 0:
        ann_return = np.nan
    else:
        ann_return = (cumulative.iloc[-1]) ** (1.0 / years) - 1
    ann_vol = returns.std(ddof=1) * np.sqrt(periods_per_year)
    sharpe = (ann_return - rf) / ann_vol if ann_vol != 0 else np.nan
    rolling_max = cumulative.cummax()
    drawdown = (rolling_max - cumulative) / rolling_max
    max_dd = drawdown.max()
    skew = stats.skew(returns.dropna())
    kurt = stats.kurtosis(returns.dropna(), fisher=False)
    negative_returns = returns[returns < 0]
    if negative_returns.empty:
        downside_dev = 0.0
        sortino = np.inf
    else:
        downside_dev = negative_returns.std(ddof=1) * np.sqrt(periods_per_year)
        sortino = (ann_return - rf) / downside_dev if downside_dev != 0 else np.nan
    stats_out = {
        'Start Date': returns.index[0].date(),
        'End Date': returns.index[-1].date(),
        'Observations': int(n),
        'Cumulative Return': float(total_return),
        'Annualized Return': float(ann_return),
        'Annualized Volatility': float(ann_vol),
        'Sharpe (rf=0)': float(sharpe),
        'Sortino (rf=0)': float(sortino),
        'Max Drawdown': float(max_dd),
        'Skewness': float(skew),
        'Kurtosis': float(kurt),
        'Periodicity per year': int(periods_per_year)
    }
    return stats_out

def build_portfolio_returns(weights_df, prices_df, indices_series_map, start_date='2015-01-01', end_date=None):
    # prices_df: DataFrame with market prices for tickers (columns uppercase)
    # indices_series_map: dict of Series for index sheets (name -> Series)
    # weights_df: DataFrame with columns Ticker, Weight
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date) if end_date is not None else None
    prices = prices_df.copy()
    tickers = list(weights_df['Ticker'].unique())
    if 'ON' in [t.upper() for t in tickers]:
        if 'ON' in indices_series_map:
            s = indices_series_map['ON']
        else:
            if len(indices_series_map) == 0:
                raise ValueError('Se solicitó ticker "ON" pero no se encontró archivo indices con series.')
            s = list(indices_series_map.values())[0]
        s = s.sort_index()
        prices = prices.copy()
        prices['ON'] = s

    # Filter by start/end
    if end is not None:
        prices = prices[(prices.index >= start) & (prices.index <= end)]
    else:
        prices = prices[prices.index >= start]
    if prices.empty:
        raise ValueError('No hay precios disponibles en el rango de fechas solicitado.')

    # --- RESAMPLE A MENSUAL (último día de mes) y calcular retornos mensuales ---
    prices = prices.sort_index()
    prices_month = prices.resample('M').last()
    returns = prices_month.pct_change()

    # Map weights and select available tickers
    weights_df = weights_df.copy()
    weights_df['Ticker'] = weights_df['Ticker'].str.upper()
    avail = [t for t in weights_df['Ticker'].tolist() if t in returns.columns]
    if len(avail) == 0:
        raise ValueError('Ningún ticker de la cartera tiene datos de precios disponibles (mensual).')
    w = weights_df.set_index('Ticker').loc[avail, 'Weight']
    r = returns[avail].copy()  # puede contener NaNs

    # calcular retorno de cartera por fila usando solo activos disponibles ese mes
    w = w.reindex(r.columns).astype(float)
    numerator = (r.fillna(0).multiply(w, axis=1)).sum(axis=1)
    denom = (r.notna().astype(float).multiply(w, axis=1)).sum(axis=1)
    port_ret = numerator.divide(denom)
    port_ret = port_ret.loc[denom > 0]
    asset_returns = r.loc[port_ret.index].copy()
    if asset_returns.empty:
        raise ValueError('Tras alinear series mensuales, no quedan observaciones comunes para la cartera.')
    return port_ret, asset_returns  # retornos mensuales y asset returns usados

def plot_portfolio_equity(port_ret, name, out_dir, start_value=10000):
    """
    Guarda un PNG con matplotlib (tema claro) y un HTML sencillo que lo muestra.
    Devuelve (html_path: Path, png_path: Path or None).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if port_ret is None or port_ret.empty:
        raise ValueError("Serie de retornos vacía para plotear.")
    equity = (1 + port_ret).cumprod() * start_value
    safe_name = "".join(c for c in name if c.isalnum() or c in (" ", "_", "-")).rstrip()
    png_path = out_dir / f"{safe_name[:50]}_equity.png"
    html_path = out_dir / f"{safe_name[:50]}_equity.html"

    # matplotlib: estilo claro para evitar fondo negro
    plt.style.use('seaborn-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(equity.index, equity.values, color='#0072BD', linewidth=1.8)
    ax.set_title(f'{name} — Equity curve (Start ${start_value:,})', fontsize=14)
    ax.set_xlabel('Date')
    ax.set_ylabel('Equity (USD)')
    ax.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.7)
    # mejorar formato fechas
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    fig.tight_layout()
    # guardar PNG
    fig.savefig(str(png_path), dpi=150, facecolor='white')
    plt.close(fig)

    # crear HTML sencillo que muestra la imagen (relativa)
    rel_png = png_path.name
    html_content = f"""<!doctype html>
<html>
<head><meta charset="utf-8"><title>{name} - Equity</title></head>
<body style="background:#ffffff;color:#000000;">
<h2>{name} — Equity curve (start ${start_value:,})</h2>
<img src="{rel_png}" alt="equity" style="max-width:100%;height:auto;border:1px solid #ccc;">
</body>
</html>"""
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f'Gráfico PNG guardado en: {png_path}')
    print(f'HTML simple guardado en: {html_path}')
    return html_path, png_path

def process_all(weights_path=None, indices_path=None, out_path=None, start_date='2016-01-01', end_date='2025-09-01'):
    base = Path(__file__).resolve().parent
    weights_path = Path(weights_path) if weights_path else base / 'weights.xlsx'
    indices_path = Path(indices_path) if indices_path else base / 'indices.xlsx'
    out_path = Path(out_path) if out_path else base / 'portfolio_stats.xlsx'

    portfolios = load_weights(weights_path)
    indices_map = {}
    if indices_path.exists():
        indices_map = load_indices(indices_path)
    else:
        # if no indices file, indices_map stays empty; code will error only if 'ON' required
        pass

    # collect all market tickers to download (excluding 'ON' handled separately)
    all_tickers = set()
    for df in portfolios.values():
        for t in df['Ticker'].tolist():
            if t.upper() != 'ON':
                all_tickers.add(t.upper())
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date) if end_date is not None else None
    market_prices = download_market_prices(sorted(all_tickers), start=start, end=end)

    charts_dir = base / 'charts'
    charts_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    # Prepare Excel writer
    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        for name, df in portfolios.items():
            try:
                port_ret, asset_returns = build_portfolio_returns(df, market_prices, indices_map, start_date=start_date, end_date=end_date)
            except Exception as e:
                # If a portfolio cannot be computed, log minimal info to summary
                summary_rows.append({
                    'Portfolio': name,
                    'Error': str(e)
                })
                print(f'Portfolio {name}: error al construir retornos: {e}')
                continue
            stats = compute_stats_from_returns(port_ret)
            stats['Portfolio'] = name
            stats['Number of Assets'] = int((df['Ticker'] != '').sum())

            # generar gráfico de equity y añadir ruta al summary
            try:
                chart_html, chart_png = plot_portfolio_equity(port_ret, name, charts_dir, start_value=10000)
                stats['Equity Chart'] = chart_html.name
                if chart_png is not None:
                    stats['Equity PNG'] = chart_png.name
                print(f'Portfolio {name}: gráficos guardados.')
            except Exception as e:
                stats['Equity Chart'] = f'plot error: {e}'
                print(f'Portfolio {name}: error al plotear: {e}')

            summary_rows.append(stats)
            # write portfolio returns and cumulative to sheet
            out_df = pd.DataFrame({
                'Portfolio Return': port_ret,
                'Portfolio Cumulative': (1 + port_ret).cumprod()
            })
            # also include asset returns used
            asset_returns.columns = [c.upper() for c in asset_returns.columns]
            out_df = out_df.join(asset_returns, how='left')
            out_df.to_excel(writer, sheet_name=name[:31])  # excel sheet name limit
        # write summary
        summary_df = pd.DataFrame(summary_rows)
        # reorder columns for readability if no error rows only
        summary_df = summary_df.set_index('Portfolio', drop=False)
        summary_df.to_excel(writer, sheet_name='Summary')
    print(f'Output escrito en: {out_path}')

if __name__ == '__main__':
    # Llamada por defecto: archivos weights.xlsx e indices.xlsx en la carpeta del script.
    process_all()