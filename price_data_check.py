#%%

import pandas as pd
import numpy as np


open_price_data_path = 'price_data/open_data.csv'
close_price_data_path = 'price_data/close_data.csv'

open_price_data = pd.read_csv(open_price_data_path, header=[0], index_col=[0], parse_dates=True, dayfirst=True)
close_price_data = pd.read_csv(close_price_data_path, header=[0], index_col=[0], parse_dates=True, dayfirst=True)

assert isinstance(open_price_data.index, pd.DatetimeIndex), "open_price_data index must be a DatetimeIndex"
assert isinstance(close_price_data.index, pd.DatetimeIndex), "close_price_data index must be a DatetimeIndex"

open_price_data = open_price_data.sort_index()
close_price_data = close_price_data.sort_index()

if not open_price_data.index.equals(close_price_data.index):
    raise ValueError("Date indices of open and close price data do not match after filtering.")

open_price_data = open_price_data.dropna(how="all")
close_price_data = close_price_data.dropna(how="all")

open_price_data = open_price_data.astype(np.float32)
close_price_data = close_price_data.astype(np.float32)

print(open_price_data.info())
print(close_price_data.info())

