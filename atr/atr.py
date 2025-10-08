import sys
import os
import pandas as pd
import yfinance as yf

TICKERS = [
    "HSBC","BK","SAN","CLS","GE","VRSN","T","KGC","SNOW","PLTR","SCHW","GILD",
    "ORCL","GLW","BBVA","SPOT","MFG","RBLX","SE","NFLX","UAL","GS","GFI","BCS",
    "TWLO","C","AVGO","NVDA","VST","JOYY","EBAY"
]

# Bajamos más días para asegurar tener datos para el cálculo (buffer)
DOWNLOAD_DAYS = "90d"
ATR_PERIOD = 25
OUT_DIR = "atr_output"

os.makedirs(OUT_DIR, exist_ok=True)

def true_range(df):
    prev_close = df["Close"].shift(1)
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - prev_close).abs()
    tr3 = (df["Low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr

def compute_atr(df, period=25):
    tr = true_range(df)
    # Wilder's smoothing (standard ATR)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    return atr

def fetch_and_compute(ticker):
    try:
        df = yf.download(ticker, period=DOWNLOAD_DAYS, interval="1d", progress=False, threads=False)
        if df.empty or "Close" not in df.columns:
            print(f"{ticker}: sin datos.")
            return None
        atr = compute_atr(df, ATR_PERIOD)
        out = df.loc[:, ["High","Low","Close"]].copy()
        out["ATR"] = atr
        out = out.dropna().tail(25)
        if out.empty:
            print(f"{ticker}: no hay suficientes datos para ATR({ATR_PERIOD}).")
            return None
        last_atr = out["ATR"].iloc[-1]
        print(f"{ticker}: ATR({ATR_PERIOD}) último = {last_atr:.6f}")
        return out
    except Exception as e:
        print(f"{ticker}: error -> {e}")
        return None

def main():
    results = {}
    for t in TICKERS:
        results[t] = fetch_and_compute(t)

    # Summary manteniendo el orden exacto de TICKERS
    summary_df = pd.DataFrame({"Ticker": TICKERS})
    summary_df["Last_ATR"] = [
        (results[t]["ATR"].iloc[-1] if results.get(t) is not None else pd.NA)
        for t in TICKERS
    ]

    # Details: concatenar las últimas 25 filas de cada ticker en el mismo Excel (una sola hoja "Details")
    details_list = []
    for t in TICKERS:
        out = results.get(t)
        if out is not None:
            tmp = out.copy().reset_index()
            # asegurar nombre de columna de fecha consistente
            tmp.rename(columns={tmp.columns[0]: "Date"}, inplace=True)
            tmp.insert(0, "Ticker", t)
            details_list.append(tmp)

    if details_list:
        details_df = pd.concat(details_list, ignore_index=True)
    else:
        details_df = pd.DataFrame(columns=["Ticker", "Date", "High", "Low", "Close", "ATR"])

    # Guardar TODO en un solo archivo Excel
    excel_path = os.path.join(OUT_DIR, "atr_all_in_one.xlsx")
    try:
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            summary_df.to_excel(writer, index=False, sheet_name="Summary")
            details_df.to_excel(writer, index=False, sheet_name="Details")
    except Exception as e:
        print("Error guardando Excel:", e)

    # También guardamos un CSV resumen por compatibilidad
    summary_df.to_csv(os.path.join(OUT_DIR, "atr_summary.csv"), index=False)
    print("\nArchivo guardado en:", os.path.abspath(excel_path))

if __name__ == "__main__":
    main()