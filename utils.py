# -*- coding: utf-8 -*-
"""
Utility functions for the momentum strategy project.
"""


import numpy as np
from strategy import run_momentum_strategy

def run_strategy_wrapper(args):
    """
    Wrapper function to run momentum strategy with given parameters.
    Returns strategy name and results for multiprocessing.

    Args:
        args (tuple): Tuple of (price_data_path, test1, test2, hold, pct_selection, stock_price_ceiling, begin, finish).

    Returns:
        tuple: (strategy_name, results_df) where strategy_name is the strategy identifier
               and results_df is the DataFrame of results or an error message.
    """
    price_data_path, test1, test2, hold, pct_selection, stock_price_ceiling, begin, finish = args
    strategy_name = f"{test1}-{test2}-{hold}-pct{pct_selection}-ceiling{stock_price_ceiling}"
    results_df = run_momentum_strategy(
        price_data_path,
        test1_period=test1,
        test2_period=test2,
        hold_period=hold,
        begin=begin,
        finish=finish,
        pct_selection=pct_selection,
        stock_price_ceiling=stock_price_ceiling
    )
    return strategy_name, results_df