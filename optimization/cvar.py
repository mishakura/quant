import yfinance as yf
import pandas as pd
import riskfolio

def download_prices(tickers, start, end):
    all_data = []
    for ticker in tickers:
        s = yf.download(ticker, start=start, end=end)["Close"]
        s.name = ticker
        all_data.append(s)
    data = pd.concat(all_data, axis=1)
    return data

def optimize_portfolio_cvar(prices):
    returns = prices.pct_change().dropna()
    port = riskfolio.Portfolio(returns=returns)
    port.assets_stats(method_mu='hist', method_cov='hist')
    model = 'Classic'  # Modelo clásico
    rm = 'ADD'         # Average Drawdown
    obj = 'MinRisk'    # Minimizar riesgo
    hist = True        # Usar escenarios históricos
    rf = 0             # Tasa libre de riesgo
    l = 0              # Sin apalancamiento
    w = port.optimization(model=model, rm=rm, obj=obj, rf=rf, l=l, hist=hist)
    return w

def export_weights_to_excel(weights, filename):
    weights.to_excel(filename)

def main():
    tickers = ["VEA","IJH","IWM","IEMG","YPF","BTC-USD","GDX","GLD","SLV","USO","URA","SPY",'ETC-USD']  # Puedes cambiar estos activos
    start = "2020-01-01"
    from datetime import datetime
    end = datetime.today().strftime('%Y-%m-%d')
    prices = download_prices(tickers, start, end)
    weights = optimize_portfolio_cvar(prices)
    export_weights_to_excel(weights, "cvar_weights.xlsx")
    print("Pesos exportados a cvar_weights.xlsx")

if __name__ == "__main__":
    main()
