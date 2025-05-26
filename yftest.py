import yfinance as yf
from datetime import datetime, timedelta

# Define the ticker
ticker = yf.Ticker("SDA")

# Define date range: 12 months back from today
info = ticker.info

# Get historical data with daily frequency
print(info)
"""GGAL: exchange: NCM
RGTI: exchange: NCM
NG: exchange: ASE
RIOT: exchange: NCM
SATL: exchange: NCM
SDA: exchange: NCM

CSNA3.SA: exchange: SAO
HAPV3.SA: exchange: SAO
JBSS3.SA: exchange: SAO
LREN3.SA: exchange: SAO
MGLU3.SA: exchange: SAO

NTCO: exchange:
PRIO3.SA: exchange: SAO
RENT3.SA: exchange: SAO



WEGE3.SA: exchange: SAO
ADS: exchange:
BBAS3.SA: exchange: SAO
BAS: exchange:
BAYN: exchange:
BBV: exchange: YHD
CS: exchange:
BSN: exchange:
DTEA: exchange:
EOAN: info error: HTTP Error 404:
FNMA: exchange: OQB
FMCC: exchange: OQB
HHPD: info error: HTTP Error 404:
NEC1.HM: exchange: HAM
NSAN: info error: HTTP Error 404:
NOKA: exchange: YHD
PKS: exchange: YHD
SMSN: exchange: YHD
TTM: exchange:
RCTB4.BA: exchange: BUE
TIIAY: exchange: PNK
TEFO: info error: HTTP Error 404:
TXR: no data
TRVV: info error: HTTP Error 404:
WBO: no data
AUY: exchange:
YZCA: info error: HTTP Error 404:"""

data = ticker.history(period="1d")  # Get the most recent trading day's data

if not data.empty:
    last_price = data["Close"].iloc[-1]
    print(f"Last closing price for GGAL: {last_price}")
else:
    print("No data available for GGAL.")