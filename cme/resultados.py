import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import yfinance as yf

# 1. Leer y procesar CPI
cpi = pd.read_excel('CPI.xlsx')
cpi.columns = [col.strip() for col in cpi.columns]

if 'Fecha' in cpi.columns and 'CPI' in cpi.columns:
    cpi['Fecha'] = pd.to_datetime(cpi['Fecha'])
    cpi = cpi.set_index('Fecha').sort_index()
    cpi = cpi[~cpi.index.duplicated(keep='last')]
    cpi = cpi.asfreq('MS')  # Asegura frecuencia mensual
    cpi = cpi.interpolate(method='linear')
else:
    raise ValueError('CPI.xlsx debe tener columnas Fecha y CPI')

# 2. Leer precios nominales
excel_file = 'indices.xlsx'
all_sheets = pd.read_excel(excel_file, sheet_name=None)
dfs = []
for name, df in all_sheets.items():
    df = df.rename(columns={df.columns[1]: name})
    df = df[['Fecha', name]]
    df = df.drop_duplicates(subset='Fecha', keep='last')
    dfs.append(df.set_index('Fecha'))

indices_df = pd.concat(dfs, axis=1)
indices_df.index = pd.to_datetime(indices_df.index)
indices_df = indices_df.sort_index()

# 2b. Descargar activos de yfinance con precios ajustados
print('\n--- DESCARGANDO ACTIVOS DE YFINANCE ---')
yf_tickers = ['SPY', 'GLD', 'BNDX', 'BND', 'VEA', 'IEMG', 'DBC', 'VFMF', 'DBMF', 'PSP', 'IPO', 'HDG']
yf_data = {}

for ticker in yf_tickers:
    try:
        print(f'Descargando {ticker}...')
        # Descargar datos desde 1947
        start_date = "1947-01-01"
        
        # Descargar con auto_adjust=True para obtener precios ya ajustados
        data = yf.download(ticker, start=start_date, progress=False, auto_adjust=True)
        if not data.empty and 'Close' in data.columns:
            yf_data[ticker] = data['Close']
            print(f'  ✓ {ticker}: {len(data)} registros (ajustado por dividendos/splits)')
        else:
            print(f'  ✗ {ticker}: Sin datos')
    except Exception as e:
        print(f'  ✗ {ticker}: Error - {e}')

# Agregar activos de yfinance ajustados (en USD)
if yf_data:
    print('\nAgregando activos de yfinance AJUSTADOS (en USD)...')
    
    # Crear una lista de DataFrames para concat
    series_list = [indices_df]
    for ticker, prices in yf_data.items():
        # Verificar si es una Serie o DataFrame y convertir apropiadamente
        if isinstance(prices, pd.Series):
            series_list.append(prices.to_frame(name=ticker))
        else:
            # Si es un DataFrame, asegurarse de que la columna tenga el nombre correcto
            df_temp = prices.copy()
            df_temp.columns = [ticker]
            series_list.append(df_temp)
    
    # Hacer merge con outer join para mantener TODAS las fechas (tanto de indices.xlsx como de yfinance)
    indices_df = pd.concat(series_list, axis=1, join='outer').sort_index()
    
    print(f'  ✓ Datos de yfinance agregados con outer join')
    print(f'  ✓ Rango de fechas extendido: {indices_df.index.min()} a {indices_df.index.max()}')
    print(f'\nTotal de activos: {len(indices_df.columns)} (locales + yfinance ajustados)')
    print(f'Total de fechas: {len(indices_df)} días')
else:
    print('\nNo se descargaron activos de yfinance')

# 3. Interpolar CPI a diario y deflactar precios por activo
# Interpolar CPI a las fechas de los índices
cpi_daily = cpi.reindex(indices_df.index, method='ffill')

print('\n--- DIAGNÓSTICO CPI ---')
print(f'CPI tiene {cpi_daily["CPI"].isna().sum()} NaN de {len(cpi_daily)} fechas')
print(f'Primeros valores de CPI:')
print(cpi_daily['CPI'].head(10))
print(f'Primeros índices nominales:')
print(indices_df.head())

# Ajustar precios nominales a reales correctamente:
# Precio_real = Precio_nominal * (CPI_base / CPI_fecha)
# Usar el primer CPI válido como base
cpi_base = cpi_daily['CPI'].dropna().iloc[0]
print(f'\nCPI base (primera fecha válida): {cpi_base}')

# Deflactar cada columna
# Todos los precios están en USD, ajustar por CPI USA excepto MEP
indices_df_real = pd.DataFrame(index=indices_df.index)

for col in indices_df.columns:
    if col == 'Mep':
        # MEP no se ajusta por CPI (es tipo de cambio, no un activo)
        print(f'\n{col}: No se ajusta por CPI (tipo de cambio)')
        indices_df_real[col] = indices_df[col]
    else:
        # Todos los demás activos están en USD, ajustar por CPI USA
        nominal = indices_df[col]
        
        # Usar el CPI de la primera fecha válida de ESTE activo como base
        first_valid_idx = nominal.first_valid_index()
        if first_valid_idx is not None:
            cpi_base_activo = cpi_daily.loc[first_valid_idx, 'CPI']
            cpi_vals = cpi_daily['CPI']
            # Crear serie real: precio nominal * (CPI_base_activo / CPI_fecha)
            real = nominal * (cpi_base_activo / cpi_vals)
            indices_df_real[col] = real
            print(f'{col}: Ajustado por CPI USA (base: {first_valid_idx.date()}, CPI={cpi_base_activo:.2f})')
        else:
            # Si no hay datos válidos, dejar como NaN
            indices_df_real[col] = nominal
            print(f'{col}: Sin datos válidos, no se ajusta')

# Crear activo sintético: Dólar Real (poder adquisitivo del USD)
# Empieza en 100 y se deflacta por CPI para mostrar pérdida de poder adquisitivo
print('\nCreando activo sintético: USD Real (poder adquisitivo del dólar)')
usd_nominal = pd.Series(100, index=indices_df.index)  # 1 USD = 100 puntos base
usd_real = usd_nominal * (cpi_base / cpi_daily['CPI'])
indices_df_real['USD Real'] = usd_real
indices_df['USD Real'] = usd_nominal  # Agregar también a nominales para consistencia

# Diagnóstico: verificar alineación y NaN por activo
print('\nNaN en precios reales por activo:')
print(indices_df_real.isna().sum())
print('\nPrimeros precios reales:')
print(indices_df_real.head(10))
print('\nÚltimos precios reales (para ver pérdida de poder adquisitivo USD):')
print(indices_df_real[['USD Real']].tail(10) if 'USD Real' in indices_df_real.columns else 'USD Real no disponible')

# Diagnóstico: comparar precios nominales, reales y CPI para las primeras fechas de cada activo
print('\n--- DIAGNÓSTICO DE AJUSTE REAL ---')
for col in indices_df.columns:
    print(f'\nActivo: {col}')
    nom = indices_df[col].dropna().head(5)
    real = indices_df_real[col].dropna().head(5)
    fechas = nom.index
    cpi_vals = cpi.reindex(fechas, method='ffill')['CPI']
    print('Fechas:')
    print(fechas)
    print('Nominal:')
    print(nom.values)
    print('CPI:')
    print(cpi_vals.values)
    print('Real:')
    print(real.values)
    print('---')

# 4. Verificar alineación de fechas
print("\nPrimeras fechas de cada índice:")
for col in indices_df_real.columns:
    print(f"{col}: {indices_df_real[col].first_valid_index()}")
print("\nÚltimas fechas de cada índice:")
for col in indices_df_real.columns:
    print(f"{col}: {indices_df_real[col].last_valid_index()}")

# 5. Calcular retornos sobre series REALES
returns = indices_df_real.pct_change(fill_method=None).dropna()
nan_count = returns.isna().sum()
zero_count = (returns == 0).sum()
print("\nCantidad de NaN en retornos diarios por índice:")
print(nan_count)
print("\nCantidad de ceros en retornos diarios por índice:")
print(zero_count)

# 6. Estadísticas relevantes básicas
stats_basic = returns.describe().T[['mean', 'std', 'min', 'max']]
stats_basic['annualized_return'] = stats_basic['mean'] * 252
stats_basic['annualized_vol'] = stats_basic['std'] * (252 ** 0.5)
print("\nEstadísticas anuales básicas:")
print(stats_basic[['annualized_return', 'annualized_vol', 'min', 'max']])

# 7. Funciones auxiliares para cálculos avanzados
def calculate_drawdown(series):
    cummax = series.cummax()
    drawdown = (series - cummax) / cummax
    return drawdown

def calculate_cagr(series):
    years = (series.index[-1] - series.index[0]).days / 365.25
    return (series[-1] / series[0]) ** (1/years) - 1

def calculate_sharpe(returns, rf=0):
    # rf: daily risk-free rate
    excess_ret = returns - rf
    return (excess_ret.mean() / excess_ret.std()) * np.sqrt(252)

# 8. Calcular todas las estadísticas solicitadas usando series REALES
dias_habiles_anio = 252
stats = []
for col in indices_df_real.columns:
    serie = indices_df_real[col].dropna()
    if len(serie) < 2:
        continue
    
    # Usar retornos logarítmicos
    retornos = np.log(serie / serie.shift(1)).dropna()
    if retornos.empty:
        continue
    
    retorno_anual = retornos.mean() * dias_habiles_anio
    std_anual = retornos.std() * np.sqrt(dias_habiles_anio)
    roll_max = serie.cummax()
    drawdown = (serie - roll_max) / roll_max
    worst_drawdown = drawdown.min()
    
    # Retorno total
    if len(serie) > 1 and serie.iloc[0] != 0:
        retorno_total = serie.iloc[-1] / serie.iloc[0] - 1
    else:
        retorno_total = np.nan
    
    # CAGR
    try:
        cagr = calculate_cagr(serie)
    except:
        cagr = np.nan
    
    skewness = retornos.skew()
    kurtosis = retornos.kurtosis()
    mejor_dia = retornos.max()
    peor_dia = retornos.min()
    fecha_mejor_dia = retornos.idxmax() if not retornos.empty else None
    fecha_peor_dia = retornos.idxmin() if not retornos.empty else None
    sharpe = retorno_anual / std_anual if std_anual != 0 else np.nan
    
    # Calcular duración del drawdown más largo
    dd = drawdown.copy()
    dd_periods = (dd < 0).astype(int)
    dd_groups = (dd_periods.diff(1) != 0).cumsum()
    dd_lengths = dd_periods.groupby(dd_groups).sum()
    max_dd_length = dd_lengths.max() / dias_habiles_anio if not dd_lengths.empty else np.nan
    
    # Drawdown actual
    last_dd = drawdown.iloc[-1]
    
    cantidad_nans = indices_df_real[col].isna().sum()
    
    # Fecha de inicio y fin de la serie
    fecha_inicio = serie.index[0]
    fecha_fin = serie.index[-1]
    
    stats.append({
        'Indice': col,
        'Fecha inicio': fecha_inicio,
        'Fecha fin': fecha_fin,
        'Retorno promedio anual': retorno_anual,
        'CAGR': cagr,
        'Desviación estándar anual': std_anual,
        'Worst drawdown': worst_drawdown,
        'Last Drawdown': last_dd,
        'Retorno total': retorno_total,
        'Skewness': skewness,
        'Kurtosis': kurtosis,
        'Mejor día': mejor_dia,
        'Fecha mejor día': fecha_mejor_dia,
        'Peor día': peor_dia,
        'Fecha peor día': fecha_peor_dia,
        'Sharpe ratio': sharpe,
        'Duración drawdown más largo (años)': max_dd_length,
        'Cantidad de datos en blanco': cantidad_nans
    })

stats_df = pd.DataFrame(stats)
print("\nEstadísticas completas (series reales):")
print(stats_df)

# 9. Graficar precios reales por activo
for col in indices_df_real.columns:
    df = indices_df_real[[col]].dropna().copy()
    if df.empty:
        print(f"No hay datos reales para graficar {col}, se omite.")
        continue
    df['Open'] = df[col].shift(1).fillna(df[col])
    df['High'] = df[[col, 'Open']].max(axis=1)
    df['Low'] = df[[col, 'Open']].min(axis=1)
    df_ohlc = df.rename(columns={col: 'Close'})[['Open', 'High', 'Low', 'Close']]
    # Convertir col a string para evitar problemas con tuplas
    col_str = str(col) if not isinstance(col, str) else col
    mpf.plot(df_ohlc, type='line', style='charles', title=f'Histórico {col_str} (ajustado por inflación)', ylabel=col_str, volume=False)

# 10. Guardar todas las estadísticas en Excel
with pd.ExcelWriter('estadisticas_indices.xlsx') as writer:
    stats_df.to_excel(writer, sheet_name='Estadisticas', index=False)
    nan_count.to_frame('NaN_count').to_excel(writer, sheet_name='NaN_count')
    zero_count.to_frame('Zero_count').to_excel(writer, sheet_name='Zero_count')
    indices_df_real.to_excel(writer, sheet_name='Series_Reales')
    indices_df.to_excel(writer, sheet_name='Series_Nominales')

print("\nEstadísticas exportadas a estadisticas_indices.xlsx")