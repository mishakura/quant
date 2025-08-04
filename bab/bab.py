import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Parameters
tickers = tickers = ["YPF","GGAL","BBAR","SUPV","PAM","TGS","TEO","IRS","CRESY","LOMA","BMA","EDN","CEPU",
        "AAL", "AAP", "AAPL", "ABBV", "ABEV", "ABNB", "ABT",
        "ACN", "ADBE", "AGRO", "ADI", "ADP", "AEM", "AMAT",
        "AMD", "AMGN", "AMX", "AMZN", "ANF", "ARCO", "ARM", "ASR",
        "AVGO", "AVY", "AZN","ASML","AI","TEAM","CLS","CEG","DECK","RGTI",
        "B", "BA", "BABA","NOW","VST","VRTX","PATH","PDD","XPEV",
         "BAK", "BB", "BHP","BRK-B",
        "BIDU", "BIIB", "BIOX", "BITF", "BKNG", "BKR", "BMY",
        "BP", "BRFS", "CAAP", "CAH",
         "CAR", "CAT", "CCL", "CDE", "CL", "COIN", "COST", "CRM",
         "CSCO", "CSNA3.SA", "CVS", "CVX", "CX", "DAL", "DD",
        "DE", "DEO", "DHR", "DIS", "DOCU", "DOW",
        "E", "EA", "EBAY", "EBR","EFX", "EQNR", "ERIC",
        "ERJ", "ETSY", "F", "FCX", "FDX",
        "FMX", "FSLR", "GE", "GFI", "GGB", "GILD",
        "GLOB", "GLW", "GM", "GOOGL", "GRMN",
        "GSK", "GT", "HAL", "HAPV3.SA", "HD", "HL", "HMC", "HMY",
        "HOG", "HON", "HPQ", "HSY", "HUT", "HWM",
         "IBM", "INFY", "INTC", "IP",
        "ISRG","JBSS3.SA",
        "JD", "JMIA", "JNJ","JOYY", "KEP", "KGC",
        "KMB", "KO", "KOD", "LAC", "LAR", "LLY",
        "LMT", "LND", "LRCX", "LREN3.SA", "LVS",  "MA", "MCD",
         "MDLZ", "MDT", "MELI", "META", "MGLU3.SA", "MMC", "MMM",
        "MO", "MOS", "MRK", "MRNA", "MRVL", "MSFT", "MSI", "MSTR",
        "MU", "MUX", "NEM", "NFLX", "NG", "NGG", "NIO",
        "NKE", "NTCO", "NTES", "NUE", "NVDA", "NVS",
        "NXE", "ORCL", "ORLY", "OXY", "PAAS", "PAC", "PAGS", "PANW", "PBI",
        "PBR", "PCAR", "PEP", "PFE", "PG",
        "PHG", "PINS", "PLTR", "PM", "PRIO3.SA", "PSX", "PYPL", "QCOM",
         "RACE", "RBLX", "RENT3.SA", "RIO", "RIOT", "ROKU", "ROST", "RTX",
         "SAP", "SATL", "SBS", "SBUX", "SCCO", "SDA", "SE",
        "SHEL", "SHOP", "SID", "SLB", "SNA", "SNAP", "SNOW", "SONY", "SPCE",
        "SPGI", "SPOT", "STLA", "STNE", "SYY", "T",
        "TCOM", "TEN", "TGT", "TIMB", "TM",
        "TMO", "TMUS", "TRIP", "TSLA", "TSM", "TTE", "TV", "TWLO",
        "TXN", "UAL", "UBER", "UGP", "UL", "UNH",
         "UNP", "URBN", "V", "VALE",
          "VIST", "VIV", "VOD", "VRSN", "VZ", "WBA", "WB", "WEGE3.SA", "WMT", "X", "XOM", "XP", "XRX",
        "XYZ", "YELP", "ZM", "ADS","AEG","AXP","AIG","BBD","BBAS3.SA","BSBR","SAN","BAC","BCS","BAS","BAYN",
        "BBV","SCHW","C","ELP","CS","BSN","DTEA","EOAN","AKO-B","FNMA", "FMCC","GPRK","HDB","HHPD",
        "HSBC","IBN","ING","IFF","JPM","ITUB","JCI","KB","LYG","MBG.F","MUFG","NEC1.HM","NSAN",
        "NOKA","NMR","PSO","PKS","SMSN","SWKS","TTM","RCTB4.BA","TIIAY","TEFO","TX","BK","GS","TRVV",
        "TJX", "USB", "UPST","WBO","WFC","AUY","YZCAY"]

today = datetime.today()
one_year_ago = today - timedelta(days=365)
five_years_ago = today - timedelta(days=5*365)

# Download data
raw_data = yf.download(tickers + [market], start=five_years_ago.strftime('%Y-%m-%d'), end=today.strftime('%Y-%m-%d'))
data = raw_data.xs('Adj Close', axis=1, level=1)

# Calculate daily log returns (for volatility, 1 year)
daily_returns = np.log(data / data.shift(1))
daily_returns_1y = daily_returns.loc[one_year_ago:]

# Calculate 3-day log returns (for correlation, 5 years)
returns_3d = np.log(data / data.shift(3))
returns_3d_5y = returns_3d.loc[five_years_ago:]

# Volatility (std of daily returns over 1 year)
volatility = daily_returns_1y[tickers].std()

# Correlation (3-day returns over 5 years, shrinkage towards 0)
corr_matrix = returns_3d_5y[tickers + [market]].corr()
shrinkage = 0.1  # Shrink correlations 10% towards zero
corr_with_market = corr_matrix.loc[tickers, market] * (1 - shrinkage)

# Beta = corr(stock, market) * (vol(stock)/vol(market))
vol_market = daily_returns_1y[market].std()
beta = corr_with_market * (volatility / vol_market)

# Rank stocks by beta (low to high)
ranking = beta.sort_values().reset_index()
ranking.columns = ['Ticker', 'Beta']
ranking['Rank'] = ranking['Beta'].rank(method='min')

# Output to Excel
ranking.to_excel('bab_beta_ranking.xlsx', index=False)
print("Excel file 'bab_beta_ranking.xlsx' created with beta rankings.")
