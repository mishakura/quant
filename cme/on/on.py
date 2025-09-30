import pandas as pd
import numpy as np


def clean_bond_prices(precios_file, cashflows_file, output_file='on_cme.xlsx'):
    precios_xls = pd.ExcelFile(precios_file)
    cashflows_xls = pd.ExcelFile(cashflows_file)
    activos = precios_xls.sheet_names
    clean_dict = {}
    pagos_dict = {}

    for activo in activos:
        # Leer precios
        df_precios = pd.read_excel(precios_file, sheet_name=activo)
        # Buscar columna de fecha (case-insensitive)
        fecha_col = [col for col in df_precios.columns if col.strip().lower() in ['fecha', 'date']][0]
        precio_col = [col for col in df_precios.columns if col.strip().lower() in ['precio', 'close', 'cierre']][0]
        df_precios[fecha_col] = pd.to_datetime(df_precios[fecha_col])
        df_precios = df_precios.sort_values(fecha_col)
        # Eliminar fechas duplicadas, quedándote con el último precio de cada fecha
        df_precios = df_precios.drop_duplicates(subset=fecha_col, keep='last')
        precios = df_precios.set_index(fecha_col)[precio_col].copy()

        # Leer cash flows
        df_cf = pd.read_excel(cashflows_file, sheet_name=activo)
        # Buscar columna de fecha de pago (case-insensitive)
        posibles_fechas = ['fecha de pago', 'fecha', 'payment date']
        fecha_pago_cols = [col for col in df_cf.columns if col.strip().lower() in posibles_fechas]
        if not fecha_pago_cols:
            print(f"[ERROR] No se encontró columna de fecha de pago en la hoja '{activo}'. Columnas disponibles: {list(df_cf.columns)}")
            continue  # Salta este activo y sigue con el resto
        fecha_pago_col = fecha_pago_cols[0]

        # Convertir a datetime, eliminar filas donde la conversión falló
        df_cf[fecha_pago_col] = pd.to_datetime(df_cf[fecha_pago_col], errors='coerce', dayfirst=True)
        df_cf = df_cf[df_cf[fecha_pago_col].notnull()]
        if df_cf.empty:
            print(f"[{activo}] No quedan filas válidas después de limpiar fechas. Se omite este activo.")
            continue
        df_cf = df_cf.sort_values(fecha_pago_col)
        # Convertir columnas a numérico plano
        df_cf['Interés'] = pd.to_numeric(df_cf['Interés'], errors='coerce').fillna(0)
        if 'Amortización' in df_cf.columns:
            df_cf['Amortización'] = pd.to_numeric(df_cf['Amortización'], errors='coerce').fillna(0)
        else:
            df_cf['Amortización'] = 0.0
        # Ajustar intereses anualizados a la periodicidad detectada
        if len(df_cf) > 1:
            fechas_pago = df_cf[fecha_pago_col]
            diffs = fechas_pago.diff().dt.days.iloc[1:]
            periodicidad_dias = diffs.mode().iloc[0] if not diffs.empty else 365
            if 27 <= periodicidad_dias <= 32:
                divisor = 12  # Mensual
            elif 85 <= periodicidad_dias <= 95:
                divisor = 4   # Trimestral
            elif 170 <= periodicidad_dias <= 190:
                divisor = 2   # Semestral
            else:
                divisor = 1   # Anual o indefinido
            df_cf['Interés'] = df_cf['Interés'] / divisor
        # Si hay columna Residual, calcular interés sobre el residual anterior
        if 'Residual' in df_cf.columns:
            residual_shifted = df_cf['Residual'].shift(1)
            residual_shifted.iloc[0] = 100.0  # O el nominal de tu bono
            df_cf['Residual_shifted'] = residual_shifted
            df_cf['Interés'] = df_cf['Interés'] * df_cf['Residual_shifted'] / 100.0
        # Calcular fecha de ajuste: 72hs hábiles antes si es antes de 2024-05-09, 48hs hábiles después
        fecha_corte = pd.Timestamp('2024-05-27')
        df_cf['Fecha ajuste'] = np.where(
            df_cf[fecha_pago_col] < fecha_corte,
            df_cf[fecha_pago_col] - pd.offsets.BDay(2),
            df_cf[fecha_pago_col] - pd.offsets.BDay(1)
        )
        # Corregir casos donde la fecha de ajuste queda posterior a la fecha de pago
        df_cf['Fecha ajuste'] = np.where(
            df_cf['Fecha ajuste'] > df_cf[fecha_pago_col],
            df_cf[fecha_pago_col] - pd.offsets.BDay(1),
            df_cf['Fecha ajuste']
        )
        # Suma acumulada de cupones y amortizaciones
        df_cf['Cumulative'] = (df_cf['Interés'] + df_cf['Amortización']).cumsum()

        # Para cada fecha de precios, sumar los pagos acumulados hasta esa fecha
        clean_series = []
        pagos_aplicados = []
        last_cumulative = 0.0
        for fecha in precios.index:
            pagos_hasta_fecha = df_cf[df_cf['Fecha ajuste'] <= fecha]
            cumulative = pagos_hasta_fecha['Cumulative'].iloc[-1] if not pagos_hasta_fecha.empty else 0.0
            clean_series.append(precios.loc[fecha] + cumulative)
            pagos_aplicados.append(cumulative - last_cumulative)
            last_cumulative = cumulative
        clean_dict[activo] = pd.Series(clean_series, index=precios.index, name=activo)
        pagos_dict[activo] = pd.Series(pagos_aplicados, index=precios.index, name=activo)

    # Unir todas las series en un DataFrame
    clean_df = pd.concat(clean_dict.values(), axis=1)
    pagos_df = pd.concat(pagos_dict.values(), axis=1)

    # Eliminar la fecha problemática de ambos DataFrames
    fecha_a_eliminar = pd.Timestamp('2025-06-05')
    if fecha_a_eliminar in clean_df.index:
        clean_df = clean_df.drop(fecha_a_eliminar)
    if fecha_a_eliminar in pagos_df.index:
        pagos_df = pagos_df.drop(fecha_a_eliminar)

    # Calcular retornos diarios, retorno promedio anual y desviación estándar anual
    stats = []
    dias_habiles_anio = 252
    dias_habiles_mes = 21
    dias_habiles_semana = 5
    for activo in clean_df.columns:
        # Serie original (puede tener NaNs al inicio y al final)
        serie_full = clean_df[activo]
        # Encontrar el primer y último índice con dato válido
        first_valid_idx = serie_full.first_valid_index()
        last_valid_idx = serie_full.last_valid_index()
        # Recortar la serie entre el primer y último dato válido
        if first_valid_idx is not None and last_valid_idx is not None:
            serie_recortada = serie_full.loc[first_valid_idx:last_valid_idx]
            cantidad_nans = serie_recortada.isna().sum()
        else:
            cantidad_nans = np.nan
        serie = serie_recortada if first_valid_idx is not None and last_valid_idx is not None else pd.Series(dtype=float)
        retornos = np.log(serie / serie.shift(1)).dropna()
        retorno_anual = retornos.mean() * dias_habiles_anio
        std_anual = retornos.std() * np.sqrt(dias_habiles_anio)
        # Worst drawdown
        roll_max = serie.cummax()
        drawdown = (serie - roll_max) / roll_max
        worst_drawdown = drawdown.min()
        # Retorno total
        retorno_total = (serie.iloc[-1] / serie.iloc[0]) - 1 if len(serie) > 1 else np.nan
        # años para duración drawdown
        años = (serie.index[-1] - serie.index[0]).days / 365.25 if len(serie) > 1 else np.nan
        # Skewness y kurtosis
        skewness = retornos.skew()
        kurtosis = retornos.kurtosis()
        # Mejor y peor día
        mejor_dia = retornos.max()
        peor_dia = retornos.min()
        # Fecha del mejor día
        fecha_mejor_dia = retornos.idxmax() if not retornos.empty else None
        # Fecha del peor día
        fecha_peor_dia = retornos.idxmin() if not retornos.empty else None
        # Ratio Sharpe y Sortino (rf=0)
        sharpe = retorno_anual / std_anual if std_anual != 0 else np.nan
        # Duración drawdown más largo
        dd = drawdown.copy()
        dd_periods = (dd < 0).astype(int)
        dd_groups = (dd_periods.diff(1) != 0).cumsum()
        dd_lengths = dd_periods.groupby(dd_groups).sum()
        max_dd_length = dd_lengths.max() / dias_habiles_anio if not dd_lengths.empty else np.nan
        stats.append({
            'Activo': activo,
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

    # Normalizar cada serie a 100 en la primera fecha disponible
    norm_df = clean_df.copy()
    for col in norm_df.columns:
        first_valid = norm_df[col].first_valid_index()
        if first_valid is not None and norm_df[col][first_valid] != 0:
            norm_df[col] = norm_df[col] / norm_df[col][first_valid] * 100

    # Agregar fila de promedio de cada estadística
    promedio = stats_df.mean(numeric_only=True)
    promedio['Activo'] = 'PROMEDIO'
    stats_df = pd.concat([stats_df, pd.DataFrame([promedio])], ignore_index=True)

    # Calcular estadísticas del índice y agregarlo a stats_df
    if not norm_df.empty:
        indice_name = 'INDICE'
        serie_indice = norm_df.mean(axis=1, skipna=True).dropna()
        # Recortar la serie entre el primer y último dato válido
        first_valid_idx = serie_indice.first_valid_index()
        last_valid_idx = serie_indice.last_valid_index()
        if first_valid_idx is not None and last_valid_idx is not None:
            serie_recortada = serie_indice.loc[first_valid_idx:last_valid_idx]
            cantidad_nans = serie_recortada.isna().sum()
        else:
            cantidad_nans = np.nan
        serie = serie_recortada if first_valid_idx is not None and last_valid_idx is not None else pd.Series(dtype=float)
        retornos = np.log(serie / serie.shift(1)).dropna()
        retorno_anual = retornos.mean() * dias_habiles_anio
        std_anual = retornos.std() * np.sqrt(dias_habiles_anio)
        roll_max = serie.cummax()
        drawdown = (serie - roll_max) / roll_max
        worst_drawdown = drawdown.min()
        retorno_total = (serie.iloc[-1] / serie.iloc[0]) - 1 if len(serie) > 1 else np.nan
        años = (serie.index[-1] - serie.index[0]).days / 365.25 if len(serie) > 1 else np.nan
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
        stats_indice = {
            'Activo': indice_name,
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
        }
        stats_df = pd.concat([stats_df, pd.DataFrame([stats_indice])], ignore_index=True)

    # Normalizar cada serie a 100 en la primera fecha disponible
    norm_df = clean_df.copy()
    for col in norm_df.columns:
        first_valid = norm_df[col].first_valid_index()
        if first_valid is not None and norm_df[col][first_valid] != 0:
            norm_df[col] = norm_df[col] / norm_df[col][first_valid] * 100

    # El índice es el promedio de los activos disponibles en cada fecha (índice arranca desde lo más viejo)
    indice = norm_df.mean(axis=1, skipna=True).dropna()
    indice_df = pd.DataFrame({'Indice': indice})

    # --- Yahoo Finance: Beta vs SPY y CEMB ---
    import yfinance as yf

    # Descargar datos desde 2015-01-01
    start_date = '2015-01-01'
    end_date = indice.index.max().strftime('%Y-%m-%d')
    spy_df = yf.download('SPY', start=start_date, end=end_date)
    cemb_df = yf.download('CEMB', start=start_date, end=end_date)
    spy = spy_df['Close'].squeeze()
    cemb = cemb_df['Close'].squeeze()

    # Alinear fechas con el índice
    df_compare = pd.concat([
        indice.rename('Indice'),
        spy.rename('SPY'),
        cemb.rename('CEMB')
    ], axis=1).dropna()

    # Calcular retornos diarios
    retornos = np.log(df_compare / df_compare.shift(1)).dropna()

    # Calcular beta del índice vs SPY y CEMB
    cov_spy = np.cov(retornos['Indice'], retornos['SPY'])[0, 1]
    var_spy = np.var(retornos['SPY'])
    beta_spy = cov_spy / var_spy if var_spy != 0 else np.nan

    cov_cemb = np.cov(retornos['Indice'], retornos['CEMB'])[0, 1]
    var_cemb = np.var(retornos['CEMB'])
    beta_cemb = cov_cemb / var_cemb if var_cemb != 0 else np.nan

    # Guardar resultados en DataFrame
    beta_df = pd.DataFrame({
        'Activo': ['SPY', 'CEMB'],
        'Beta vs Indice': [beta_spy, beta_cemb]
    })

    # Guardar todo en Excel
    with pd.ExcelWriter(output_file) as writer:
        clean_df.to_excel(writer, sheet_name='clean')
        pagos_df.to_excel(writer, sheet_name='pagos_aplicados')
        stats_df.to_excel(writer, sheet_name='stats', index=False)
        indice_df.to_excel(writer, sheet_name='indice')
        beta_df.to_excel(writer, sheet_name='beta_vs_yahoo', index=False)

    print(f"Precios clean exportados a {output_file}")

# Ejemplo de uso:
clean_bond_prices('on_precios.xlsx', 'on_cashflows.xlsx')