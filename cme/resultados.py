import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
import numpy as np

# Leer todas las hojas del Excel
excel_file = 'indices.xlsx'
all_sheets = pd.read_excel(excel_file, sheet_name=None)

# Unir todas las hojas en un solo DataFrame
dfs = []
for name, df in all_sheets.items():
    df = df.rename(columns={df.columns[1]: name})  # Asume columnas: Fecha, Precio
    df = df[['Fecha', name]]
    df = df.drop_duplicates(subset='Fecha', keep='last')  # <-- Agrega esta línea
    dfs.append(df.set_index('Fecha'))

indices_df = pd.concat(dfs, axis=1)
indices_df.index = pd.to_datetime(indices_df.index)
indices_df = indices_df.sort_index()

# 1. Revisar NaN y ceros en retornos diarios
returns = indices_df.pct_change().dropna()
nan_count = returns.isna().sum()
zero_count = (returns == 0).sum()
print("Cantidad de NaN en retornos diarios por índice:")
print(nan_count)
print("\nCantidad de ceros en retornos diarios por índice:")
print(zero_count)

# 2. Correlación con retornos semanales y mensuales
returns_w = indices_df.resample('W').last().pct_change().dropna()
returns_m = indices_df.resample('M').last().pct_change().dropna()
corr_daily = returns.corr()
corr_weekly = returns_w.corr()
corr_monthly = returns_m.corr()
print("\nCorrelación de retornos diarios:")
print(corr_daily)
print("\nCorrelación de retornos semanales:")
print(corr_weekly)
print("\nCorrelación de retornos mensuales:")
print(corr_monthly)

# 3. Verificar alineación de fechas
print("\nPrimeras fechas de cada índice:")
for col in indices_df.columns:
    print(f"{col}: {indices_df[col].first_valid_index()}")
print("\nÚltimas fechas de cada índice:")
for col in indices_df.columns:
    print(f"{col}: {indices_df[col].last_valid_index()}")

# 4. Graficar precios y retornos para detectar outliers o saltos raros
fig, axes = plt.subplots(len(indices_df.columns), 2, figsize=(14, 4*len(indices_df.columns)))
for i, col in enumerate(indices_df.columns):
    axes[i, 0].plot(indices_df.index, indices_df[col])
    axes[i, 0].set_title(f'Precio {col}')
    axes[i, 1].plot(returns.index, returns[col])
    axes[i, 1].set_title(f'Retorno diario {col}')
plt.tight_layout()
plt.show()

# 5. Estadísticas relevantes
stats = returns.describe().T[['mean', 'std', 'min', 'max']]
stats['annualized_return'] = stats['mean'] * 252
stats['annualized_vol'] = stats['std'] * (252 ** 0.5)
print("\nEstadísticas anuales:")
print(stats[['annualized_return', 'annualized_vol', 'min', 'max']])

# 6. Correlación rolling de 60 días
rolling_corr = returns.rolling(60).corr()
print("\nCorrelación rolling 60 días (ejemplo entre los dos primeros índices):")
if len(indices_df.columns) >= 2:
    print(rolling_corr.iloc[rolling_corr.index.get_level_values(1)==indices_df.columns[1], 0])

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

# Calcular métricas adicionales
extra_stats = []
for col in indices_df.columns:
    price = indices_df[col].dropna()
    ret = returns[col].dropna()
    if len(price) > 1 and len(ret) > 1:
        cagr = calculate_cagr(price)
        sharpe = calculate_sharpe(ret)
        drawdown = calculate_drawdown(price)
        max_dd = drawdown.min()
        last_dd = drawdown.iloc[-1]
        extra_stats.append({
            'Indice': col,
            'CAGR': cagr,
            'Sharpe': sharpe,
            'Max_Drawdown': max_dd,
            'Last_Drawdown': last_dd
        })
extra_stats_df = pd.DataFrame(extra_stats).set_index('Indice')
print("\nEstadísticas avanzadas:")
print(extra_stats_df)

# Calcular todas las estadísticas solicitadas para cada índice
dias_habiles_anio = 252
stats = []
for col in indices_df.columns:
    serie = indices_df[col].dropna()
    retornos = np.log(serie / serie.shift(1)).dropna()
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
    skewness = retornos.skew()
    kurtosis = retornos.kurtosis()
    mejor_dia = retornos.max()
    peor_dia = retornos.min()
    fecha_mejor_dia = retornos.idxmax() if not retornos.empty else None
    fecha_peor_dia = retornos.idxmin() if not retornos.empty else None
    sharpe = retorno_anual / std_anual if std_anual != 0 else np.nan
    dd = drawdown.copy()
    dd_periods = (dd < 0).astype(int)
    dd_groups = (dd_periods.diff(1) != 0).cumsum()
    dd_lengths = dd_periods.groupby(dd_groups).sum()
    max_dd_length = dd_lengths.max() / dias_habiles_anio if not dd_lengths.empty else np.nan
    cantidad_nans = serie.isna().sum()
    stats.append({
        'Indice': col,
        'Retorno promedio anual': retorno_anual,
        'Desviación estándar anual': std_anual,
        'Worst drawdown': worst_drawdown,
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

# Guardar todas las estadísticas en una sola hoja
with pd.ExcelWriter('estadisticas_indices.xlsx') as writer:
    stats_df.to_excel(writer, sheet_name='Estadisticas', index=False)
    corr_daily.to_excel(writer, sheet_name='Correlacion_Diaria')
    corr_weekly.to_excel(writer, sheet_name='Correlacion_Semanal')
    corr_monthly.to_excel(writer, sheet_name='Correlacion_Mensual')
    nan_count.to_frame('NaN_count').to_excel(writer, sheet_name='NaN_count')
    zero_count.to_frame('Zero_count').to_excel(writer, sheet_name='Zero_count')
    # Rolling correlation entre los dos primeros índices
    if len(indices_df.columns) >= 2:
        rolling_corr_pair = rolling_corr.xs(indices_df.columns[1], level=1)[indices_df.columns[0]]
        rolling_corr_pair.to_frame('RollingCorr').to_excel(writer, sheet_name='RollingCorr')

print("Estadísticas exportadas a estadisticas_indices.xlsx")