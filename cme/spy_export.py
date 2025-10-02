import yfinance as yf
import pandas as pd
from datetime import datetime

def export_spy_data():
    """
    Descarga y exporta datos históricos de SPY (S&P 500 ETF)
    """
    
    # Descargar datos de SPY
    print("Descargando datos de SPY...")
    spy = yf.Ticker("SPY")
    
    # Obtener datos históricos (últimos 5 años por defecto)
    df = spy.history(period="5y")
    
    # Mostrar información básica
    print(f"\nDatos descargados: {len(df)} registros")
    print(f"Rango de fechas: {df.index[0].strftime('%Y-%m-%d')} a {df.index[-1].strftime('%Y-%m-%d')}")
    print("\nPrimeras filas:")
    print(df.head())
    
    # Exportar a CSV
    csv_filename = f"spy_data_{datetime.now().strftime('%Y%m%d')}.csv"
    df.to_csv(csv_filename)
    print(f"\nDatos exportados a: {csv_filename}")
    
    # Exportar a Excel
    excel_filename = f"spy_data_{datetime.now().strftime('%Y%m%d')}.xlsx"
    df.to_excel(excel_filename, sheet_name="SPY")
    print(f"Datos exportados a: {excel_filename}")
    
    # Obtener información adicional
    print("\n" + "="*50)
    print("INFORMACIÓN ADICIONAL DE SPY")
    print("="*50)
    
    info = spy.info
    print(f"Nombre: {info.get('longName', 'N/A')}")
    print(f"Precio actual: ${info.get('currentPrice', 'N/A')}")
    print(f"Volumen promedio: {info.get('averageVolume', 'N/A'):,}")
    print(f"Market Cap: ${info.get('marketCap', 'N/A'):,}")
    print(f"Dividend Yield: {info.get('dividendYield', 0)*100:.2f}%")
    
    # Calcular estadísticas
    print("\n" + "="*50)
    print("ESTADÍSTICAS")
    print("="*50)
    print(f"Precio máximo (5 años): ${df['High'].max():.2f}")
    print(f"Precio mínimo (5 años): ${df['Low'].min():.2f}")
    print(f"Retorno total: {((df['Close'][-1] / df['Close'][0]) - 1) * 100:.2f}%")
    print(f"Volatilidad anualizada: {df['Close'].pct_change().std() * (252**0.5) * 100:.2f}%")
    
    return df

if __name__ == "__main__":
    try:
        data = export_spy_data()
    except Exception as e:
        print(f"\nError: {e}")
        print("Asegúrate de tener instalado yfinance: pip install yfinance")
