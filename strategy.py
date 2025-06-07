#%%

import numpy as np
import pandas as pd
from datetime import datetime
import math
from strategy_utils import calculate_consecutive_overlap, calculate_period_results

def run_momentum_strategy(price_data_path, test1_period, test2_period, hold_period, begin = datetime(2019, 3, 28).date(), finish = datetime(2025, 3, 28).date(), pct_selection=0.5, stock_price_ceiling=None):
    """
    Runs the momentum strategy for relative strength momentum trading.

    Args:
        price_data_path (str): Path to the price data CSV file.
        test1_period (int): Number of days for the first test period.
        test2_period (int): Number of days for the second test period.
        hold_period (int): Number of days for the hold period.
        begin (datetime.date): Start date for the strategy.
        finish (datetime.date): End date for the strategy.
        pct_selection (float): Percentage of outperforming stocks to select.
        stock_price_ceiling (float, optional): Maximum stock price for inclusion.

    Returns:
        pandas.DataFrame: DataFrame containing strategy results with consecutive overlap analysis,
                         or None if the strategy fails due to insufficient data or errors.
    """
    # Load and format price data
    price_data = pd.read_csv(price_data_path, header=[0,1], index_col=[0], parse_dates=True)
    price_data.index = pd.to_datetime(price_data.index, dayfirst=True).date
    price_data = price_data.dropna(how="all")

    # Sort and filter data
    price_data = price_data[(price_data.index >= begin) & (price_data.index <= finish)].sort_index()
    available_dates = price_data.index.tolist()

    # Check if there are enough dates
    min_required_dates = (test1_period + test2_period + hold_period) - 1
    if len(available_dates) < min_required_dates:
        return None

    # Initialize result DataFrame and variables
    result_df = pd.DataFrame()
    ProcessID = 1
    start_idx = 0

    while True:
        # Check if we have enough data left
        if start_idx + min_required_dates >= len(available_dates):
            break
        
        # Define period indices
        start_t1_idx = start_idx
        end_t1_idx = start_t1_idx + (test1_period - 1)

        start_t2_idx = end_t1_idx + 1
        end_t2_idx = start_t2_idx + (test2_period - 1)

        start_hold_idx = end_t2_idx + 1
        end_hold_idx = start_hold_idx + (hold_period - 1)

        # Get actual dates
        start_t1 = available_dates[start_t1_idx]
        end_t1 = available_dates[end_t1_idx]

        start_t2 = available_dates[start_t2_idx]
        end_t2 = available_dates[end_t2_idx]

        start_hold = available_dates[start_hold_idx]
        end_hold = available_dates[end_hold_idx]

        # Run strategy periods
        t1_results = calculate_period_results(
            price_data=price_data,
            start_date=start_t1,
            end_date=end_t1,
            pct_selection=pct_selection,
            hold=False,
            test1=True,
            test2=False
        )
        if 'error' in t1_results:
            return f'error in t1, ProcessID: {ProcessID}, start_t1: {start_t1}, end_t1: {end_t1}'
        
        t2_results = calculate_period_results(
            price_data=price_data,
            start_date=start_t2,
            end_date=end_t2,
            eligible_stocks=t1_results['selected_stocks'],
            hold=False,
            test1=False,
            test2=True
        )
        if 'error' in t2_results:
            return f'error in t2, ProcessID: {ProcessID}, start_t2: {start_t2}, end_t2: {end_t2}'
        
        hold_results = calculate_period_results(
            price_data=price_data,
            start_date=start_hold,
            end_date=end_hold,
            eligible_stocks=t2_results['selected_stocks'],
            hold=True,
            end_t2=end_t2,
            start_hold=start_hold,
            hold_period=hold_period,
            stock_price_ceiling=stock_price_ceiling,
            test1=False,
            test2=False
        )
        if 'error' in hold_results:
            return f'error in hold, ProcessID: {ProcessID}, start_hold: {start_hold}, end_hold: {end_hold}'
        
        # Compile results
        result_dict = {
            'strategyID': f"{test1_period}-{test2_period}-{hold_period}",
            'pct_selection': pct_selection,
            'ProcessID': ProcessID,
#            'test_1_stocks': t1_results['num_selected_stocks'],
            'holding_stocks': hold_results['num_selected_stocks'],
#            't1_to_t2_pass_%': t2_results['num_selected_stocks'] / t1_results['num_selected_stocks'] if t1_results['num_selected_stocks'] > 0 else 0,
#            't2_to_hold_pass_%': len(hold_results['outp_stocks']) / t2_results['num_selected_stocks'] if t2_results['num_selected_stocks'] > 0 else 0,
#            'stocks': ','.join(hold_results['selected_stocks']),
            'start_date': hold_results['start_date'],
            'end_date': hold_results['end_date'],
            'avg_return': hold_results['avg_return'],
            'portfolio_return': hold_results['portfolio_return'],
            'outperformance': hold_results['outperformance'],
#            'total_stocks': hold_results['total_stocks'],
            'total_minimum_investment_amount': hold_results['total_minimum_investment_amount'],
#            'trading_costs': hold_results['trading_costs'],
#            'num_upper_circuit_stocks': hold_results['num_upper_circuit_stocks'],
#            'num_lower_circuit_stocks': hold_results['num_lower_circuit_stocks'],
#            '%_of_positive_stocks': hold_results['%_of_positive_stocks'],
#            '%_of_negative_stocks': hold_results['%_of_negative_stocks'],
#            'avg_positive_return': hold_results['avg_positive_return'],
#            'avg_negative_return': hold_results['avg_negative_return'],
#            'upper_circuit_stocks': ','.join(hold_results['upper_circuit_stocks']),
#            'lower_circuit_stocks': ','.join(hold_results['lower_circuit_stocks']),
            'stock_price_ceiling': stock_price_ceiling
        }

        # Update index and ProcessID
        start_idx = start_idx + hold_period
        ProcessID += 1
    
        # Append result to DataFrame
        result_df = pd.concat([result_df, pd.DataFrame([result_dict])], ignore_index=True)

    # Appending columns in DataFrame
    if not result_df.empty:
        result_df['portfolio_cum_return'] = (1 + result_df['portfolio_return']).cumprod()
        result_df['avg_cum_return'] = (1 + result_df['avg_return']).cumprod()
        result = result_df

        # Calculate consecutive overlap
#        result = calculate_consecutive_overlap(result_df, stocks_col='stocks', max_back_steps=5, drop_temp=True)
    else:
        result = result_df

    # Save and print results
    if result is not None:
        pass
    else:
        print("Strategy failed due to insufficient data or errors.")

    return result

# Example usage
if __name__ == "__main__":
    price_data_path = r"E:\Internship_Hashbrown\self\Relative Strength Momentum Strategy\Source\CMIE_Price_Data_Cleaned.csv"
    test1_period = 10
    test2_period = 10
    hold_period = 10
    begin = datetime(2005, 1, 3).date()
    finish = datetime(2024,12, 31).date()
    stock_price_ceiling = None

    results_df = run_momentum_strategy(price_data_path, test1_period, test2_period, hold_period,begin, finish, pct_selection= 1,stock_price_ceiling = stock_price_ceiling)
    results_df.to_csv('test_strategy_results.csv', index=False)
    print(results_df)