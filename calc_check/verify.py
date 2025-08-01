#%%

import pandas as pd
import numpy as np

price_data_path = 'momentum_test_data.csv'

price_data = pd.read_csv(price_data_path, header=[0,1], index_col=[0], parse_dates=True)

