import numpy as np
import openpyxl
import pandas as pd

# Leer el archivo Excel
input_file = 'bd_data.xlsx'
df = pd.read_excel(input_file)
ticker_ratio = {'MMM': 10.0, 'ABT': 4.0, 'ABBV': 10.0, 'ANF': 1.0, 'ACN': 75.0, 'ADGO': 1.0, 'ADS': 22.0, 'ADBE': 44.0, 'JMIA': 1.0, 'AAP': 14.0, 'AMD': 10.0, 'AEG': 1.0, 'AEM': 6.0, 'ABNB': 15.0, 'BABA': 9.0, 'GOOGL': 58.0, 'AABA': 3.0, 'MO': 4.0, 'AOCA': 1.0, 'AMZN': 144.0, 'ABEV': 0.3333333333333333, 'AMX': 1.0, 'AAL': 2.0, 'AXP': 15.0, 'AIG': 5.0, 'AMGN': 30.0, 'ADI': 15.0, 'AAPL': 20.0, 'AMAT': 5.0, 'ARCO': 0.5, 'ARKK': 10.0, 'ARM': 27.0, 'AZN': 2.0, 'T': 3.0, 'ADP': 6.0, 'AVY': 18.0, 'CAR': 26.0, 'BIDU': 11.0, 'BKR': 7.0, 'BBD': 1.0, 'BBAS3': 2.0, 'BSBR': 1.0, 'SAN': 0.25, 'BA.C': 4.0, 'BCS': 1.0, 'GOLD': 2.0, 'BAS': 2.0, 'BAYN': 3.0, 'BRKB': 22.0, 'BHP': 2.0, 'BBV': 1.0, 'BIOX': 1.0, 'BIIB': 13.0, 'BITF': 0.2, 'BB': 3.0, 'BKNG': 700.0, 'BP': 5.0, 'LND': 1.0, 'BAK': 2.0, 'BRFS': 0.3333333333333333, 'BMY': 3.0, 'AVGO': 39.0, 'BNG': 5.0, 'CAJ': 2.0, 'CAH': 3.0, 'CCL': 3.0, 'CAT': 20.0, 'CX': 1.0, 'EBR': 0.25, 'SCHW': 13.0, 'CVX': 16.0, 'LFC': 2.0, 'SNP': 3.0, 'CSCO': 5.0, 'C': 3.0, 'KOFM': 2.0, 'CDE': 1.0, 'COIN': 27.0, 'CL': 3.0, 'CBRD': 1.0, 'SBS': 0.5, 'ELP': 0.3333333333333333, 'SID': 0.125, 'GLW': 4.0, 'CAAP': 0.25, 'COST': 48.0, 'CS': 1.0, 'CVS': 15.0, 'DHR': 54.0, 'BSN': 20.0, 'DE': 40.0, 'DAL': 8.0, 'DESP': 1.0, 'DTEA': 3.0, 'DEO': 6.0, 'DOCU': 22.0, 'DOW': 6.0, 'DD': 5.0, 'EOAN': 6.0, 'EBAY': 2.0, 'EA': 14.0, 'LLY': 56.0, 'AKO.B': 1.0, 'ERJ': 1.0, 'XLE': 2.0, 'E': 4.0, 'EFX': 16.0, 'EQNR': 6.0, 'GLD': 50.0, 'ETSY': 16.0, 'XOM': 10.0, 'FNMA': 1.0, 'FDX': 10.0, 'RACE': 83.0, 'XLF': 2.0, 'FSLR': 18.0, 'FMX': 6.0, 'F': 1.0, 'FMCC': 1.0, 'FCX': 3.0, 'GRMN': 3.0, 'GE': 8.0, 'GM': 6.0, 'GPRK': 1.0, 'GGB': 0.25, 'GILD': 4.0, 'GLOB': 18.0, 'GFI': 1.0, 'GT': 2.0, 'PAC': 16.0, 'ASR': 20.0, 'TV': 3.0, 'GSK': 4.0, 'HAL': 2.0, 'HAPV3': 1.0, 'HOG': 3.0, 'HMY': 1.0, 'HDB': 2.0, 'HL': 1.0, 'HHPD': 2.0, 'HMC': 1.0, 'HON': 8.0, 'HWM': 1.0, 'HPQ': 1.0, 'HSBC': 2.0, 'HNPIY': 1.0, 'HUT': 0.2, 'IBN': 1.0, 'INFY': 1.0, 'ING': 3.0, 'INTC': 5.0, 'IBM': 15.0, 'IFF': 12.0, 'IP': 4.0, 'ISRG': 90.0, 'QQQ': 20.0, 'IBIT': 10.0, 'FXI': 5.0, 'IEUR': 11.0, 'ETHA': 5.0, 'EWZ': 2.0, 'EEM': 5.0, 'IBB': 27.0, 'IVW': 20.0, 'IVE': 40.0, 'IWM': 10.0, 'ITUB': 1.0, 'JPM': 15.0, 'JD': 4.0, 'JNJ': 15.0, 'JCI': 2.0, 'YY': 5.0, 'KB': 2.0, 'KMB': 6.0, 'KGC': 1.0, 'PHG': 5.0, 'KEP': 1.0, 'LRCX': 56.0, 'LVS': 2.0, 'LAAC': 1.0, 'LAC': 1.0, 'LYG': 2.0, 'ERIC': 2.0, 'RENT3': 2.0, 'LMT': 20.0, 'LREN3': 1.0, 'MGLU3': 1.0, 'MMC': 16.0, 'MRVL': 14.0, 'MA': 33.0, 'MCD': 24.0, 'MUX': 2.0, 'MDT': 4.0, 'MELI': 120.0, 'MBG': 4.0, 'MRK': 5.0, 'META': 24.0, 'MU': 5.0, 'MSFT': 30.0, 'MSTR': 20.0, 'MUFG': 1.0, 'MFG': 1.0, 'MBT': 2.0, 'MRNA': 19.0, 'MDLZ': 15.0, 'MSI': 20.0, 'NGG': 2.0, 'NTCO': 1.0, 'NEC1': 0.3333333333333333, 'NTES': 14.0, 'NFLX': 48.0, 'NEM': 3.0, 'NXE': 1.0, 'NKE': 12.0, 'NIO': 4.0, 'NSAN': 1.0, 'NOKA': 1.0, 'NMR': 1.0, 'NG': 0.25, 'NVS': 4.0, 'NLM': 2.0, 'UN': 2.0, 'NUE': 16.0, 'NVDA': 24.0, 'OXY': 5.0, 'ORCL': 3.0, 'ORAN': 1.0, 'ORLY': 222.0, 'PCAR': 3.0, 'PAGS': 3.0, 'PLTR': 3.0, 'PANW': 50.0, 'PAAS': 3.0, 'PCRF': 2.0, 'PYPL': 8.0, 'PSO': 1.0, 'PEP': 18.0, 'PRIO3': 2.0, 'PBR': 1.0, 'PTR': 4.0, 'PFE': 4.0, 'PM': 18.0, 'PSX': 6.0, 'PINS': 7.0, 'PBI': 1.0, 'OGZD': 2.0, 'LKOD': 4.0, 'ATAD': 4.0, 'PKS': 3.0, 'PG': 15.0, 'SH': 8.0, 'QCOM': 11.0, 'RTX': 5.0, 'RIO': 8.0, 'RIOT': 3.0, 'RBLX': 2.0, 'ROKU': 13.0, 'ROST': 4.0, 'SHEL': 2.0, 'SPGI': 45.0, 'CRM': 18.0, 'SMSN': 14.0, 'SAP': 6.0, 'SATL': 1.0, 'SLB': 3.0, 'SE': 32.0, 'SHPW': 0.5, 'SHOP': 107.0, 'SIEGY': 3.0, 'SI': 10.0, 'SWKS': 21.0, 'SNAP': 1.0, 'SNA': 6.0, 'SNOW': 30.0, 'SONY': 8.0, 'SCCO': 2.0, 'DIA': 20.0, 'SPY': 20.0, 'SPOT': 28.0, 'SQ': 20.0, 'SBUX': 12.0, 'STLA': 5.0, 'STNE': 3.0, 'SDA': 2.0, 'SUZ': 1.0, 'SYY': 8.0, 'TSM': 9.0, 'TGT': 24.0, 'TTM': 1.0, 'RCTB4': 0.001, 'TIIAY': 1.0, 'VIV': 1.0, 'TEFO': 8.0, 'TEN': 1.0, 'TXR': 4.0, 'TSLA': 15.0, 'TXN': 5.0, 'BK': 2.0, 'BA': 24.0, 'KO': 5.0, 'XLC': 19.0, 'XLY': 43.0, 'XLP': 16.0, 'GS': 13.0, 'XLV': 29.0, 'HSY': 21.0, 'HD': 32.0, 'XLI': 28.0, 'XLB': 18.0, 'MOS': 5.0, 'XLRE': 9.0, 'XLK': 46.0, 'TRVV': 6.0, 'DISN': 12.0, 'TMO': 22.0, 'TIMB': 1.0, 'TJX': 22.0, 'TMUS': 33.0, 'TTE': 3.0, 'TM': 15.0, 'TCOM': 2.0, 'TRIP': 2.0, 'TWLO': 36.0, 'TWTR': 2.0, 'USB': 5.0, 'UBER': 2.0, 'UGP': 1.0, 'UL': 3.0, 'UNP': 20.0, 'UAL': 5.0, 'X': 3.0, 'UNH': 33.0, 'UPST': 5.0, 'URBN': 2.0, 'VALE': 2.0, 'VEA': 10.0, 'VRSN': 6.0, 'VZ': 4.0, 'SPCE': 0.5, 'V': 18.0, 'VIST': 3.0, 'VOD': 1.0, 'WBA': 3.0, 'WMT': 18.0, 'WBO': 6.0, 'WFC': 5.0, 'XROX': 1.0, 'XP': 4.0, 'AUY': 1.0, 'YZCA': 2.0, 'YELP': 2.0, 'ZM': 47.0, 'ASML': 146.0, 'TEAM': 47.0, 'AI': 5.0, 'CLS': 20.0, 'CEG': 45.0, 'DECK': 25.0, 'RGTI': 2.0, 'PPD': 25.0, 'NOW': 172.0, 'TEM': 12.0, 'PATH': 2.0, 'VRTX': 101.0, 'VST': 26.0, 'XPEV': 4.0, 'PSQ': 8.0, 'VIG': 39.0, 'SLV': 6.0, 'IJH': 12.0, 'EWJ': 14.0}
# Seleccionar columnas relevantes
cols = ['SIMBOLO', 'FECHA', 'CIERRE']
df = df[cols]

# Pivotear la tabla: fechas como índice, activos como columnas, cierre como valor
pivot_df = df.pivot_table(index='FECHA', columns='SIMBOLO', values='CIERRE', aggfunc='last')

# Resetear el índice para que FECHA sea una columna
pivot_df = pivot_df.reset_index()

# Crear el nuevo activo FX = AL30 / AL30D
if 'AL30' in pivot_df.columns and 'AL30D' in pivot_df.columns:
    pivot_df['FX'] = pivot_df['AL30'] / pivot_df['AL30D']

# Convertir precios a USD: solo divide por FX si el símbolo NO termina en D o C

# Crear un DataFrame para la hoja USD con los nombres originales

# Construir todas las series en un diccionario para evitar fragmentación
usd_data = {'FECHA': pivot_df['FECHA']}
fx_numeric = pd.to_numeric(pivot_df['FX'], errors='coerce')
for col in pivot_df.columns:
    if col in ['FECHA', 'FX']:
        continue
    price_numeric = pd.to_numeric(pivot_df[col], errors='coerce')
    if col in ['GLD', 'YPFD']:
        usd_data[col] = price_numeric / fx_numeric
    elif col.endswith('D') or col.endswith('C'):
        usd_data[col] = price_numeric
    else:
        usd_data[col] = price_numeric / fx_numeric
usd_df = pd.DataFrame(usd_data)

# Multiplicar cada columna por su ratio si está en ticker_ratio
for col in usd_df.columns:
    if col == 'FECHA':
        continue
    if col in ticker_ratio:
        usd_df[col] = usd_df[col] * ticker_ratio[col]

# Normalizar series dirty a clean usando cash flows de rentas.xlsx
def get_accrued_interest(cashflows, date):
    # Encuentra el último pago antes de la fecha
    pagos = cashflows[cashflows['Fecha de pago'] <= date]
    if pagos.empty:
        return 0.0
    last_payment = pagos.iloc[-1]
    # Encuentra el próximo pago después de la fecha
    futuros = cashflows[cashflows['Fecha de pago'] > date]
    if futuros.empty:
        return 0.0
    next_payment = futuros.iloc[0]
    # Días entre pagos (periodicidad)
    days_total = (next_payment['Fecha de pago'] - last_payment['Fecha de pago']).days
    days_since = (date - last_payment['Fecha de pago']).days
    if days_total == 0:
        return 0.0
    # Interés anual del próximo pago
    interes_anual = next_payment['Interés'] / 100  # Convertir a decimal
    # Ajustar interés anual a la periodicidad
    interes_periodo = interes_anual * (days_total / 365)
    # Interés devengado proporcional
    accrued = interes_periodo * (days_since / days_total)
    # SUMAR el interés devengado
    return accrued

try:
    rentas_xls = pd.ExcelFile('rentas.xlsx')
    fechas = pd.to_datetime(usd_df['FECHA'])  # Convertir fechas a Timestamps
    for col in usd_df.columns:
        if col == 'FECHA':
            continue
        if col in rentas_xls.sheet_names:
            cashflows = pd.read_excel(rentas_xls, sheet_name=col)
            # Limpiar símbolo % y convertir columna 'Interés' a numérico
            cashflows['Interés'] = cashflows['Interés'].astype(str).str.replace('%', '').str.replace(',', '.')
            cashflows['Interés'] = pd.to_numeric(cashflows['Interés'], errors='coerce').fillna(0)
            # Ajustar intereses anualizados a la periodicidad detectada ANTES de la suma acumulada
            if len(cashflows) > 1:
                cashflows = cashflows.sort_values('Fecha de pago')
                fechas_pago = pd.to_datetime(cashflows['Fecha de pago'])
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
                cashflows['Interés'] = cashflows['Interés'] / divisor
            # Ordenar cashflows por fecha de pago por si acaso
            cashflows = cashflows.sort_values('Fecha de pago')
            # Inicializar una columna auxiliar para la suma acumulada
            clean_prices = usd_df[col].copy()
            # Crear una serie de fechas de pago menos 1 día (24hs antes)
            cashflows['Fecha ajuste'] = pd.to_datetime(cashflows['Fecha de pago']) - pd.Timedelta(days=1)
            # Ordenar por fecha de ajuste
            cashflows = cashflows.sort_values('Fecha ajuste')
            # Filtrar cashflows para solo incluir pagos a partir de la primera fecha de la serie
            fecha_inicio = pd.to_datetime(usd_df['FECHA']).min()
            cashflows = cashflows[cashflows['Fecha ajuste'] >= fecha_inicio]
            # Calcular suma acumulada de intereses y amortizaciones
            cashflows['Interés'] = cashflows['Interés'].fillna(0)
            if 'Amortización' in cashflows.columns:
                cashflows['Amortización'] = pd.to_numeric(cashflows['Amortización'].astype(str).str.replace('%', '').str.replace(',', '.'), errors='coerce').fillna(0)
                cashflows['Amortización'] = cashflows['Amortización'] * 100  # Multiplicar por 100 según requerimiento
                cashflows['Cumulative'] = cashflows['Interés'].cumsum() + cashflows['Amortización'].cumsum()
            else:
                cashflows['Cumulative'] = cashflows['Interés'].cumsum()
            # Para cada fecha en el time series, sumar la suma acumulada de cupones y amortizaciones pagados hasta esa fecha
            fechas = pd.to_datetime(usd_df['FECHA'])
            cumulative_coupons = []
            for f in fechas:
                pagos_hasta_fecha = cashflows[cashflows['Fecha ajuste'] <= f]
                if not pagos_hasta_fecha.empty:
                    cumulative = pagos_hasta_fecha['Cumulative'].iloc[-1]
                else:
                    cumulative = 0.0
                cumulative_coupons.append(cumulative)
            clean_prices = clean_prices + cumulative_coupons
            usd_df[col] = clean_prices
except Exception as e:
    print('Error al procesar rentas.xlsx:', e)
# Guardar el resultado en un nuevo archivo Excel

# Guardar el resultado en un nuevo archivo Excel con dos hojas: original y USD
output_file = 'bd_data_merged.xlsx'
with pd.ExcelWriter(output_file) as writer:
    pivot_df.to_excel(writer, index=False, sheet_name='Original')
    usd_df.to_excel(writer, index=False, sheet_name='USD')
