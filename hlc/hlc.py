import yfinance as yf
import pandas as pd

# Descargar datos diarios de GOOGL
df = yf.download("GOOGL", period="10y", interval="1d")
df = df[['Close']]
df = df.dropna()

# Resampleos (usando frecuencias recomendadas)
df['W_Close'] = df['Close'].resample('W-FRI').last().reindex(df.index, method='ffill')
df['M_Close'] = df['Close'].resample('M').last().reindex(df.index, method='ffill')
df['Q_Close'] = df['Close'].resample('4M').last().reindex(df.index, method='ffill')  # Cuatrimestral

# Shift para obtener los cierres anteriores
df['W_Close_prev'] = df['W_Close'].shift(1)
df['M_Close_prev'] = df['M_Close'].shift(1)
df['Q_Close_prev'] = df['Q_Close'].shift(1)

# Backtest
trades = []
in_position = False

for i in range(2, len(df)):
    row = df.iloc[i]
    prev_row = df.iloc[i-1]
    # Condiciones de entrada
    cond_long = (
        row['W_Close'] > row['W_Close_prev'] and
        row['M_Close'] > row['M_Close_prev'] and
        row['Q_Close'] > row['Q_Close_prev']
    )
    # Entrada: el precio del día anterior llegó al W2 (W_Close_prev)
    touched_w2 = abs(prev_row['Close'] - row['W_Close_prev']) < 1e-6 or prev_row['Close'] <= row['W_Close_prev'] + 1e-6

    if not in_position and cond_long and touched_w2:
        entry_price = row['Close']
        entry_date = df.index[i]
        stop_loss = entry_price * 0.95
        hold_days = 21  # Aproximadamente 1 mes de trading
        in_position = True

        # Buscar salida
        exit_price = None
        exit_date = None
        for j in range(i+1, min(i+hold_days+1, len(df))):
            future_row = df.iloc[j]
            # Stop loss
            if future_row['Close'] <= stop_loss:
                exit_price = future_row['Close']
                exit_date = df.index[j]
                break
        else:
            # Si no tocó stop, salir al final del mes
            exit_price = df.iloc[min(i+hold_days, len(df)-1)]['Close']
            exit_date = df.index[min(i+hold_days, len(df)-1)]

        trades.append({
            'entry_date': entry_date,
            'entry_price': entry_price,
            'exit_date': exit_date,
            'exit_price': exit_price,
            'result_%': (exit_price - entry_price) / entry_price * 100
        })
        in_position = False

# Guardar resultados en CSV
trades_df = pd.DataFrame(trades)
trades_df.to_csv('hlc_trades_debug.csv', index=False)
print(trades_df)