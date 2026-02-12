import pandas as pd
import numpy as np

def read_cashflows(excel_file):
    xls = pd.ExcelFile(excel_file)
    activos = xls.sheet_names
    # Diccionario: activo -> DataFrame
    cashflows = {}
    for activo in activos:
        df = pd.read_excel(excel_file, sheet_name=activo)
        # Buscar columna de fecha ignorando mayúsculas/minúsculas y espacios
        fecha_col = None
        for col in df.columns:
            if str(col).strip().lower().replace(' ', '') == 'fechadepago':
                fecha_col = col
                break
        if fecha_col is None:
            raise ValueError(f"No se encontró columna de fecha en la hoja '{activo}'")
        df[fecha_col] = pd.to_datetime(df[fecha_col])
        # Buscar columnas relevantes ignorando mayúsculas/tildes/espacios
        def buscar_col(df, nombre):
            nombre = nombre.lower().replace('í', 'i').replace('é', 'e').replace('á', 'a').replace('ó', 'o').replace('ú', 'u').replace(' ', '')
            for col in df.columns:
                col_norm = str(col).lower().replace('í', 'i').replace('é', 'e').replace('á', 'a').replace('ó', 'o').replace('ú', 'u').replace(' ', '')
                if col_norm == nombre:
                    return col
            return None

        # Usar Interés + Amortización como cash flow
        for nombre in ['Interés', 'Amortización']:
            col = buscar_col(df, nombre)
            if col is None:
                df[nombre] = 0
        df['Flujo'] = df[[buscar_col(df, 'Interés'), buscar_col(df, 'Amortización')]].sum(axis=1)
        # Agrupa por mes
        df['Mes'] = df[fecha_col].dt.to_period('M')
        mensual = df.groupby('Mes')['Flujo'].sum()
        cashflows[activo] = mensual
    # Construye matriz: filas = meses, columnas = activos
    meses = sorted(set(m for cf in cashflows.values() for m in cf.index))
    matriz = pd.DataFrame(index=meses, columns=activos).fillna(0)
    for activo, serie in cashflows.items():
        matriz.loc[serie.index, activo] = serie.values
    return matriz

def main():
    excel_file = input("Ingrese el nombre del archivo Excel de cashflows: ")
    matriz = read_cashflows(excel_file)
    activos = matriz.columns

    weights_file = input("Ingrese el nombre del archivo Excel de weights (por ejemplo, weights.xlsx): ")
    df_weights = pd.read_excel(weights_file)
    # Normaliza nombres de columnas
    df_weights.columns = [str(c).strip().lower() for c in df_weights.columns]
    # Extrae datos
    tickers = df_weights['ticker'].tolist()
    weights = df_weights['weight'].tolist()
    precios = df_weights['precio'].tolist()
    precios_dict = dict(zip(tickers, precios))

    # Verifica que los tickers coincidan
    if set(tickers) != set(activos):
        print("\nAdvertencia: los tickers en weights.xlsx no coinciden con los activos en el archivo de cashflows.")
        print("Tickers en cashflows:", list(activos))
        print("Tickers en weights:", tickers)

    suma_ponderaciones = sum(weights)
    if abs(suma_ponderaciones - 1) > 1e-6:
        print(f"\nAdvertencia: la suma de ponderaciones es {suma_ponderaciones:.4f} (debe ser 1).")

    monto_total = float(input("\nIngrese el monto total a invertir (por ejemplo, 10000): "))

    cantidades = {activo: (weights[i] * monto_total) / precios_dict[activo] for i, activo in enumerate(tickers)}

    flujo_mensual_total = np.zeros(len(matriz.index))
    for i, activo in enumerate(tickers):
        flujo_unitario = np.array(matriz[activo].values, dtype=float)
        flujo_mensual_total += flujo_unitario * cantidades[activo]

    print("\nCash flow mensual resultante para la inversión:")
    for mes, flujo in zip(matriz.index, flujo_mensual_total):
        print(f"{mes}: {flujo:.2f}")

    promedio = np.mean(flujo_mensual_total)
    print(f"\nPromedio mensual recibido: {promedio:.2f}")

    df_resultado = pd.DataFrame({
        'Mes': [str(m) for m in matriz.index],
        'Cash Flow Mensual': flujo_mensual_total
    })
    df_resultado.to_excel('cashflow_resultado.xlsx', index=False)
    print("\nCash flow mensual exportado a 'cashflow_resultado.xlsx'.")

if __name__ == "__main__":
    main()