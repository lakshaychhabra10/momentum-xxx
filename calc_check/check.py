#%%

import pandas as pd

# Load into DataFrame
df = pd.read_excel('Book1.xlsx', sheet_name='Sheet1')
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%y')
df = df.set_index('Date')

# Drop NaNs
prices = df['3M India Ltd.'].dropna()

# --- Buy and Hold Return ---
buy_and_hold_return = prices.iloc[-1] / prices.iloc[0] - 1

# --- 2-Day Rolling Returns (compounded) ---
chunk_returns = []
for i in range(0, len(prices) - 1, 2):  # step=2 for non-overlapping 2-day windows
    start = prices.iloc[i]
    end = prices.iloc[i + 1]
    ret = end / start - 1
    chunk_returns.append(ret)

# Cumulative return from 2-day periods
cumulative_return = 1
for r in chunk_returns:
    cumulative_return *= (1 + r)
cumulative_return -= 1

# --- Output ---
print(f"Buy-and-Hold Return     : {buy_and_hold_return:.4f}")
print(f"2-Day Chunks Return     : {cumulative_return:.4f}")
print(f"Difference              : {abs(buy_and_hold_return - cumulative_return):.4f}")
