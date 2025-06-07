#%%

from itertools import product

# min_range = 10
# max_range = 100

test1_range = [10,15,20,25,30,35,40,45,50,55,60,65,70,80,85,90,95,100]  # Test1 period range
test2_range = [10,15,20,25,30,35,40,45,50,55,60,65,70,80,85,90,95,100]  # Test1 period range
hold_range = [10,15,20,25,30,35,40,45,50,55,60,65,70,80,85,90,95,100]  # Test1 period range

# test1_range = range(min_period, max_period + 1)
# test2_range = range(min_period, max_period + 1)
# hold_range = range(min_period, max_period + 1)

pct_selection = [0.05, 0.1, 0.2, 0.3, 0.4, 1]  # Percentage of stocks to select
stock_price_ceiling = [1200, 1800, 500000]  # Price ceiling for stock selection

# Generate all combinations
combinations = list(product(test1_range, test2_range, hold_range, pct_selection, stock_price_ceiling))
len(combinations)
# combinations = [(10,10,10,1,1200)]