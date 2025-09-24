## Seleccionar aquellas Ons que tengan la mayor cantidad de data que sean de calidad crediticia como IRSA, YPF, TELECOM, VIST, PAMPA.
## Juntar toda esa data en un excel con su historial de precios
## Tener otro excel con su historial de cash flow para normalizar la serie y que sea clean
## Computar la desviación estándar de cada serie
## Computar el promedio de todas las series de desviación estándar.
## Calcular retorno promedio también.

## YMCID, TLC1D,YCA6O, YCAMO. BYC2D


##Datos de , BYC2D, YCA6O e YCAMO desde 2017 hasta 2022 (5 años)
#TLC1D desde 2019 hasta 2025 (6 años)
##
## Luego hay que usar otros datos.

import pandas as pd
import numpy as np
import pyxirr as xirr  # asegúrate de tener esto al inicio

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
        fecha_corte = pd.Timestamp('2024-05-09')
        df_cf['Fecha ajuste'] = np.where(
            df_cf[fecha_pago_col] < fecha_corte,
            df_cf[fecha_pago_col] - pd.offsets.BDay(3),
            df_cf[fecha_pago_col] - pd.offsets.BDay(2)
        )
        # Suma acumulada de cupones y amortizaciones
        df_cf['Cumulative'] = df_cf['Interés'].cumsum() + df_cf['Amortización'].cumsum()

        # Para cada fecha de precios, sumar los pagos acumulados hasta esa fecha
        clean_series = []
        pagos_aplicados = []
        last_cumulative = 0.0
        for fecha in precios.index:
            pagos_hasta_fecha = df_cf[df_cf['Fecha ajuste'] <= fecha]
            if not pagos_hasta_fecha.empty:
                cumulative = pagos_hasta_fecha['Cumulative'].iloc[-1]
            else:
                cumulative = 0.0
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
    for activo in clean_df.columns:
        serie = clean_df[activo].dropna()
        retornos = np.log(serie / serie.shift(1)).dropna()
        retorno_anual = retornos.mean() * dias_habiles_anio
        std_anual = retornos.std() * np.sqrt(dias_habiles_anio)

        # Yield simple y TIR (YTM) solo con flujos futuros
        yield_simple = np.nan
        tir = np.nan
        try:
            df_cf = pd.read_excel(cashflows_file, sheet_name=activo)
            posibles_fechas = ['fecha de pago', 'fecha', 'payment date']
            fecha_pago_cols = [col for col in df_cf.columns if col.strip().lower() in posibles_fechas]
            if fecha_pago_cols:
                fecha_pago_col = fecha_pago_cols[0]
                df_cf[fecha_pago_col] = pd.to_datetime(df_cf[fecha_pago_col], errors='coerce', dayfirst=True)
                df_cf = df_cf[df_cf[fecha_pago_col].notnull()]
                first_valid = serie.first_valid_index()
                if first_valid is not None:
                    # Solo flujos futuros (>= primer precio)
                    df_cf_fut = df_cf[df_cf[fecha_pago_col] >= first_valid].copy()
                    if not df_cf_fut.empty:
                        intereses = pd.to_numeric(df_cf_fut['Interés'], errors='coerce').fillna(0)
                        amort = pd.to_numeric(df_cf_fut['Amortización'], errors='coerce').fillna(0) if 'Amortización' in df_cf_fut.columns else 0
                        total_cupones = intereses.sum() + amort.sum()
                        precio_inicial = serie.loc[first_valid]
                        last_pago = df_cf_fut[fecha_pago_col].max()
                        años = (last_pago - first_valid).days / 365.25 if last_pago and first_valid else np.nan
                        yield_simple = (total_cupones / precio_inicial) / años if años and años > 0 else np.nan

                        # TIR/YTM usando np.irr sobre los flujos filtrados
                        flujos = [-precio_inicial] + list(intereses + amort)
                        if len(flujos) > 1:
                            tir = np.irr(flujos)
                            if tir is not None and np.isfinite(tir):
                                tir = tir
                            else:
                                tir = np.nan
        except Exception as e:
            yield_simple = np.nan
            tir = np.nan

        stats.append({
            'Activo': activo,
            'Retorno promedio anual': retorno_anual,
            'Desviación estándar anual': std_anual,
            'Yield simple anualizado': yield_simple,
            'TIR (YTM)': tir
        })
    stats_df = pd.DataFrame(stats)

    # Normalizar cada serie a 100 en la primera fecha disponible
    norm_df = clean_df.copy()
    for col in norm_df.columns:
        first_valid = norm_df[col].first_valid_index()
        if first_valid is not None and norm_df[col][first_valid] != 0:
            norm_df[col] = norm_df[col] / norm_df[col][first_valid] * 100

    # El índice es el promedio de los activos disponibles en cada fecha (índice arranca desde lo más viejo)
    indice = norm_df.mean(axis=1, skipna=True).dropna()
    indice_df = pd.DataFrame({'Indice': indice})

    # Cargar precios dirty
    dirty_xls = pd.ExcelFile('on_precios.xlsx')
    dirty_dict = {}
    for activo in dirty_xls.sheet_names:
        df_dirty = pd.read_excel(dirty_xls, sheet_name=activo)
        fecha_col = [col for col in df_dirty.columns if col.strip().lower() in ['fecha', 'date']][0]
        precio_col = [col for col in df_dirty.columns if col.strip().lower() in ['precio', 'close', 'cierre']][0]
        df_dirty[fecha_col] = pd.to_datetime(df_dirty[fecha_col])
        df_dirty = df_dirty.drop_duplicates(subset=fecha_col, keep='last')
        dirty_dict[activo] = df_dirty.set_index(fecha_col)[precio_col]
    dirty_df = pd.DataFrame(dirty_dict)

    # Calcular TIR diaria usando precios dirty
    tir_diaria_dict = {}
    for activo in dirty_df.columns:
        tir_diaria = []
        precios_serie = dirty_df[activo]
        # Leer cashflow original
        df_cf = pd.read_excel(cashflows_file, sheet_name=activo)
        posibles_fechas = ['fecha de pago', 'fecha', 'payment date']
        fecha_pago_cols = [col for col in df_cf.columns if col.strip().lower() in posibles_fechas]
        if not fecha_pago_cols:
            tir_diaria_dict[activo] = pd.Series([np.nan]*len(precios_serie), index=precios_serie.index)
            continue
        fecha_pago_col = fecha_pago_cols[0]
        df_cf[fecha_pago_col] = pd.to_datetime(df_cf[fecha_pago_col], errors='coerce', dayfirst=True)
        df_cf = df_cf[df_cf[fecha_pago_col].notnull()]
        intereses = pd.to_numeric(df_cf['Interés'], errors='coerce').fillna(0)
        amort = pd.to_numeric(df_cf['Amortización'], errors='coerce').fillna(0) if 'Amortización' in df_cf.columns else 0
        pagos = intereses + amort
        fechas_pagos = df_cf[fecha_pago_col]
        for fecha_actual in precios_serie.index:
            precio_actual = precios_serie.loc[fecha_actual]
            # Flujos futuros desde fecha_actual
            mask_fut = fechas_pagos >= fecha_actual
            flujos_futuros = pagos[mask_fut].values
            fechas_futuras = fechas_pagos[mask_fut].values
            if len(flujos_futuros) == 0 or pd.isna(precio_actual) or precio_actual == 0:
                tir_diaria.append(np.nan)
                continue
            # Construir los flujos y fechas para xirr
            cashflows = [-precio_actual] + list(flujos_futuros)
            cashflow_dates = [fecha_actual] + list(fechas_futuras)
            try:
                tir = xirr.xirr(dict(zip(cashflow_dates, cashflows)))
                tir_diaria.append(tir)
            except Exception:
                tir_diaria.append(np.nan)
        tir_diaria_dict[activo] = pd.Series(tir_diaria, index=precios_serie.index)

    tir_diaria_df = pd.DataFrame(tir_diaria_dict)

    with pd.ExcelWriter(output_file) as writer:
        clean_df.to_excel(writer, sheet_name='clean')
        pagos_df.to_excel(writer, sheet_name='pagos_aplicados')
        stats_df.to_excel(writer, sheet_name='stats', index=False)
        indice_df.to_excel(writer, sheet_name='indice')
        tir_diaria_df.to_excel(writer, sheet_name='tir_diaria')

    print(f"Precios clean exportados a {output_file}")

# Ejemplo de uso:
clean_bond_prices('on_precios.xlsx', 'on_cashflows.xlsx')


