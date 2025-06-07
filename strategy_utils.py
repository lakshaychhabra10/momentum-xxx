#%%

import numpy as np
import pandas as pd
from datetime import datetime
import math
from datetime import date


def calculate_equated_investment_plan(stock_prices: pd.Series):
    """
    Calculates an equated investment plan given a Series of stock prices.

    The plan is to invest an equal amount in each stock, by buying shares of each stock
    until the total investment in that stock is equal to the highest stock price.

    Parameters
    ----------
    stock_prices : pd.Series
        Pandas Series mapping stock symbols to their prices.

    Returns
    -------
    dict
        Dictionary containing the total investment required, a DataFrame of the stock plan,
        a Series mapping stock symbols to number of shares to buy, and a dictionary mapping
        stock symbols to the investment amount for each stock.
    """

    # Validate input
    if not isinstance(stock_prices, pd.Series) or stock_prices.empty:
        raise ValueError("Invalid input: stock_prices must be a non-empty pandas Series.")
    
    if (stock_prices <= 0).any():
        raise ValueError("Invalid stock prices: all prices must be positive.")

    # Calculate the highest stock price
    highest_stock_price = stock_prices.max()

    total_investment_required = 0
    stock_plan = {}
    investment_mapping = {}
    num_shares_mapping = {}

    # This works with pd.Series as it implements .items() method
    # pd.Series.items() returns (index, value) pairs just like dict.items()
    for stock, price in stock_prices.items():
        shares_to_buy = round(highest_stock_price / price)
        investment = shares_to_buy * price

        stock_plan[stock] = {
            "price_per_share": price,
            "shares_to_buy": shares_to_buy,
            "investment": investment
        }

        num_shares_mapping[stock] = shares_to_buy
        investment_mapping[stock] = investment
        total_investment_required += investment

    return {
        "total_investment": total_investment_required,
        "stock_plan": pd.DataFrame.from_dict(stock_plan, orient='index'),
        "num_shares_mapping": pd.Series(num_shares_mapping),
        "investment_per_stock": investment_mapping
    }

def calculate_trading_cost(buy_val_per_stock: pd.Series, sell_val_per_stock: pd.Series, holding_period: int) -> float:

    """
    Calculates the total trading cost for a given set of stocks.

    Args:
        buy_val_per_stock (pd.Series): Series of buy values per stock.
        sell_val_per_stock (pd.Series): Series of sell values per stock.
        holding_period (int): Number of days for which the stocks are held.

    Returns:
        float: Total trading cost.

    Notes:
        The total cost is calculated as the sum of brokerage, STT, exchange transaction charges, SEBI charges, stamp duty, and GST.
        For intraday transactions (holding_period = 1), brokerage is calculated as a percentage of the total value of the trade.
        For delivery transactions (holding_period > 1), brokerage is 0.
    """

    # Error handling
    if not isinstance(buy_val_per_stock, pd.Series):
        raise ValueError("buy_val_per_stock must be a pandas Series.")
    if not isinstance(sell_val_per_stock, pd.Series):
        raise ValueError("sell_val_per_stock must be a pandas Series.")
    if not buy_val_per_stock.index.equals(sell_val_per_stock.index):
        raise ValueError("Indexes of buy_val_per_stock and sell_val_per_stock must match.")
    if not isinstance(holding_period, int) or holding_period < 1:
        raise ValueError("holding_period must be a positive integer (>= 1).")

    total_cost = 0

    for stock in buy_val_per_stock.index:
        buy_val = buy_val_per_stock.at[stock]
        sell_val = sell_val_per_stock.at[stock]

        if holding_period == 1:
            # Intraday (MIS)

            brokerage = min(0.03/100 * (buy_val + sell_val), 20)
            stt = 0.025/100 * sell_val
            exch_txn_charges = 0.002997/100 * (buy_val + sell_val)
            sebi_charges = 0.0001/100 * (buy_val + sell_val)
            stamp_duty = 0.003/100 * buy_val
            gst = 18/100 * (brokerage + exch_txn_charges + sebi_charges)

            cost = int(brokerage + stt + exch_txn_charges + sebi_charges + stamp_duty + gst)

        else:
            # Delivery (CNC)

            brokerage = 0
            stt = 0.1/100 * (buy_val + sell_val)
            exch_txn_charges = 0.002997/100 * (buy_val + sell_val)
            sebi_charges = 0.0001/100 * (buy_val + sell_val)
            stamp_duty = 0.015/100 * buy_val
            gst = 18/100 * (brokerage + exch_txn_charges + sebi_charges)

            cost = int(brokerage + stt + exch_txn_charges + sebi_charges + stamp_duty + gst)

        total_cost += cost

    return total_cost


def calculate_weighted_portfolio_return(buy_val_per_stock : pd.Series ,sell_val_per_stock : pd.Series , trading_costs : float , returns : pd.Series):

    """
    Calculates the weighted portfolio return for a given set of stocks.

    Parameters
    ----------
    buy_val_per_stock : pd.Series
        Series of buy values per stock.
    sell_val_per_stock : pd.Series
        Series of sell values per stock.
    trading_costs : function
        Function to calculate trading costs.
    returns : pd.Series
        Series of individual stock returns.

    Returns
    -------
    dict
        Dictionary containing the weighted return and individual returns.
    """

    total_buy_value = buy_val_per_stock.sum()
    total_sell_value = sell_val_per_stock.sum()

    weighted_return = ( total_sell_value - total_buy_value - trading_costs ) / total_buy_value

    individual_returns = {}

    for stock in buy_val_per_stock.index:
        individual_returns[stock] = returns[stock]
    
    return {
        "weighted_return": weighted_return,
        "individual_returns": pd.Series(individual_returns),
    }


def upper_circuit_tester(close_price_data, open_price_data, returns, open_date, stocks):
    """
    Tests if specified stocks opened at their upper circuit limit in the Indian stock market.
    
    Args:
        close_price_data (pd.Series): Previous day's closing prices (stocks as index).
        open_price_data (pd.Series): Current day's opening prices (stocks as index).
        returns (pd.Series): Pre-calculated returns ((open - close) / close), same index as inputs.
        close_date (pd.Timestamp or datetime): Date of the closing prices.
        open_date (pd.Timestamp or datetime): Date of the opening prices.
        stocks (list): List of stock names to test for upper circuit.
    
    Returns:
        pd.Series: Boolean Series indicating if each specified stock opened at upper circuit (True = cannot buy).
    """
    # Filter the input Series to only include the specified stocks
    close_price_data = close_price_data[close_price_data.index.isin(stocks)]
    open_price_data = open_price_data[open_price_data.index.isin(stocks)]
    returns = returns[returns.index.isin(stocks)]
    
    # Define upper circuit percentages
    circuit_limits = [0.02, 0.05, 0.10, 0.15, 0.20]
    
    # Determine tick size rules based on open_date
    june_2024 = datetime(2024, 6, 1).date()
    threshold = 250 if open_date >= june_2024 else 100
    tick_low = 0.01
    tick_high = 0.05
    
    # Initialize result Series with False, only for the specified stocks
    result = pd.Series(False, index=close_price_data.index)
    
    # Function to round down to the nearest tick size
    def round_down_to_tick(price, tick_size):
        return (price // tick_size) * tick_size
    
    # Test each circuit limit
    for circuit in circuit_limits:
        # Calculate theoretical upper circuit price based on close price
        upper_circuit_price = close_price_data * (1 + circuit)
        
        # Apply tick size rounding
        tick_size = np.where(upper_circuit_price < threshold, tick_low, tick_high)
        upper_circuit_price = pd.Series(
            [round_down_to_tick(price, ts) for price, ts in zip(upper_circuit_price, tick_size)],
            index=upper_circuit_price.index
        )
        
        # Calculate theoretical return at upper circuit
        theoretical_return = (upper_circuit_price - close_price_data) / close_price_data
        
        # Compare actual return with theoretical return
        # Within -0.1% difference (i.e., actual return >= theoretical return - 0.001)
        diff = returns - theoretical_return
        is_upper_circuit = (diff >= -0.0005) & (diff <= 0)  # Actual return should be slightly below or equal
        
        # Update result where this circuit limit applies
        result = result | is_upper_circuit
    
    return result



def lower_circuit_tester(close_price_data, open_price_data, returns, open_date, stocks):
    """
    Tests if specified stocks opened at their lower circuit limit in the Indian stock market.
    
    Args:
        close_price_data (pd.Series): Previous day's closing prices (stocks as index).
        open_price_data (pd.Series): Current day's opening prices (stocks as index).
        returns (pd.Series): Pre-calculated returns ((open - close) / close), same index as inputs.
        close_date (pd.Timestamp or datetime): Date of the closing prices.
        open_date (pd.Timestamp or datetime.date): Date of the opening prices.
        stocks (list): List of stock names to test for lower circuit.
    
    Returns:
        pd.Series: Boolean Series indicating if each specified stock opened at lower circuit (True = hit lower circuit).
    """
    # Filter the input Series to only include the specified stocks
    close_price_data = close_price_data[close_price_data.index.isin(stocks)]
    open_price_data = open_price_data[open_price_data.index.isin(stocks)]
    returns = returns[returns.index.isin(stocks)]
    
    # Define lower circuit percentages
    circuit_limits = [0.02, 0.05, 0.10, 0.20]
    
    # Define june_2024 as a datetime.date object
    june_2024 = datetime(2024, 6, 1).date()
    
    # Determine tick size rules based on open_date
    threshold = 250 if open_date >= june_2024 else 100
    tick_low = 0.01
    tick_high = 0.05
    
    # Initialize result Series with False, only for the specified stocks
    result = pd.Series(False, index=close_price_data.index)
    
    # Function to round up to the nearest tick size
    def round_up_to_tick(price, tick_size):
        return -((-price // tick_size) * tick_size)
    
    # Test each circuit limit
    for circuit in circuit_limits:
        # Calculate theoretical lower circuit price based on close price
        lower_circuit_price = close_price_data * (1 - circuit)
        
        # Apply tick size rounding (round up)
        tick_size = np.where(lower_circuit_price < threshold, tick_low, tick_high)
        lower_circuit_price = pd.Series(
            [round(round_up_to_tick(price, ts),2) for price, ts in zip(lower_circuit_price, tick_size)],
            index=lower_circuit_price.index
        )
        
        # Calculate theoretical return at lower circuit
        theoretical_return = (lower_circuit_price - close_price_data) / close_price_data
        
        # Compare actual return with theoretical return
        # Within +0.1% difference (i.e., actual return <= theoretical return + 0.001)
        diff = returns - theoretical_return
        is_lower_circuit = (diff <= 0.001) & (diff >= 0)  # Actual return should be slightly above or equal
        
        # Update result where this circuit limit applies
        result = result | is_lower_circuit
    
    return result



def calculate_consecutive_overlap(df, stocks_col='stocks', max_back_steps=5, drop_temp=True):
    """
    Calculate percentage of stocks in the current row that were present in all of the previous N rows consecutively.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with stock information.
    stocks_col : str, optional
        Column containing comma-separated stock symbols.
    max_back_steps : int, optional
        Maximum number of past rows to check consecutively (default: 10).
    drop_temp : bool, optional
        Whether to drop intermediate columns (default: True).
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with added overlap columns.
    """
    df = df.copy()
    df['_stocks_list'] = df[stocks_col].str.split(',')
    df['_stocks_set'] = df['_stocks_list'].apply(set)

    for step in range(1, max_back_steps + 1):
        col_name = f'pct_overlap_with_prev_{step}'
        df[col_name] = 0.0

        for i in range(step, len(df)):
            current_set = df.at[i, '_stocks_set']
            
            # Initialize with the set from the first previous row
            intersect_set = df.at[i - 1, '_stocks_set']
            
            # Intersect with all previous rows up to step
            for j in range(2, step + 1):
                intersect_set = intersect_set.intersection(df.at[i - j, '_stocks_set'])
            
            # Final intersection with current row
            overlap_set = current_set.intersection(intersect_set)
            pct_overlap = len(overlap_set) / len(current_set) if current_set else 0
            df.at[i, col_name] = pct_overlap

    if drop_temp:
        df = df.drop(['_stocks_list', '_stocks_set'], axis=1)

    return df


def calculate_period_results(price_data, start_date, end_date,  pct_selection = 1,  eligible_stocks=None, hold=False, end_t2 = None , start_hold = None, hold_period = None, stock_price_ceiling = None, test1 = False, test2 = False):
    """
    Calculates results for a given period.

    Args:
        start_date (datetime.date): Start date of the period.
        end_date (datetime.date): End date of the period.
        eligible_stocks (list[str], optional): List of eligible stocks.
        hold (bool): Whether to perform hold period calculations.

        global variables:
        stock_price_ceiling (float): Price ceiling for stock selection.
        pct_selection (float): Percentage of stocks to select based on performance.
        start_hold (datetime.date): Start date for hold period calculations.
        end_t2 (datetime.date): End date for previous period calculations.
        hold_period (int): Number of days for which the stocks are held.

    Returns:
        dict: Dictionary containing results for the period.
    """
    try:
        # Get opening prices for start_date and closing prices for end_date
        start_prices = price_data.loc[start_date]['Open'] 
        end_prices = price_data.loc[end_date]['Close']  

        # Calculate returns
        # Align both start and end prices by index, then drop any stock with missing values
        combined_prices = pd.concat([start_prices, end_prices], axis=1, keys=['Open', 'Close']).dropna()
        # Calculate returns only on stocks with complete data
        returns = (combined_prices['Close'] / combined_prices['Open'] - 1)

        # Calculate average return and get total stocks
        avg_return = returns.mean()
        total_stocks = returns[returns.notna()].index.tolist()

        # Select stocks based on eligible_stocks parameter

        if test1:
            outperforming_returns = returns[returns >= avg_return]
            selected_stocks = outperforming_returns.nlargest(int(math.ceil(len(outperforming_returns) * pct_selection))).index.tolist() 
            selected_return = returns[selected_stocks].mean()

            # Initialize result dictionary
            result = {
                'selected_stocks': selected_stocks,
                'num_selected_stocks': len(selected_stocks),
                'universe_return': avg_return,
                'selected_return': selected_return,
                'total_stocks': len(total_stocks)
            }
        
        if test2:
            if eligible_stocks:
                eligible_stocks = [stock for stock in eligible_stocks if stock in combined_prices.index]
                filtered_returns = returns[eligible_stocks]
                selected_stocks = filtered_returns[filtered_returns >= avg_return].index.tolist()
                selected_return = filtered_returns[filtered_returns >= avg_return].mean()

            # Initialize result dictionary
            result = {
                'selected_stocks': selected_stocks,
                'num_selected_stocks': len(selected_stocks),
                'universe_return': avg_return,
                'eligible_stocks': eligible_stocks,
                'filtered_returns': filtered_returns,
                'selected_return': selected_return,
                'total_stocks': len(total_stocks)
            }
        
        if hold:

            temp_result = {
            'start_date': start_date,
            'end_date': end_date,
            'universe_return': avg_return,
            'total_stocks': len(total_stocks),
            'selected_stocks': [],
            'num_selected_stocks': 0,
            'portfolio_return': 0,
            'outperformance': -(avg_return),
            'weighted_portfolio_return': 0,
            'total_minimum_investment_amount': 0,
            'trading_costs': 0,
            'num_upper_circuit_stocks': 0,
            'upper_circuit_stocks': [],
            'num_lower_circuit_stocks': 0,
            'lower_circuit_stocks': [],
            '%_of_positive_stocks': 0,
            '%_of_negative_stocks': 0,
            'avg_positive_return': 0,
            'avg_negative_return': 0,
            'stock_price_ceiling': stock_price_ceiling,
            }

            if not eligible_stocks:
                return temp_result
            
            if eligible_stocks:
                eligible_stocks = [stock for stock in eligible_stocks if stock in combined_prices.index]
                filtered_returns = returns[eligible_stocks]

                # Circuit testing
                prev_close_data = price_data.loc[end_t2]['Close']   
                current_open_data = price_data.loc[start_hold]['Open']  

                combined_open_prev_close_prices = pd.concat([prev_close_data, current_open_data], axis=1, keys=['PrevClose', 'Open']).dropna()

                # Update indices to only include valid (non-NaN) stocks
                valid_index = combined_open_prev_close_prices.index

                prev_close_data = prev_close_data.loc[valid_index]
                current_open_data = current_open_data.loc[valid_index]


                # Filter stocks based on price ceiling
                # Option 1: Using list comprehension with a conditional check
                if stock_price_ceiling:
                    filtered_stocks = [stock for stock in eligible_stocks 
                                    if stock in current_open_data and current_open_data[stock] < stock_price_ceiling]
                else:
                    filtered_stocks = eligible_stocks

                # If no stocks are left after filtering, return early with temp_result
                if not filtered_stocks:
                    temp_result.update({
                        'num_upper_circuit_stocks': 0,
                        'upper_circuit_stocks': [],
                        'num_lower_circuit_stocks': 0,
                        'lower_circuit_stocks': [],
                    })
                    return temp_result

                # Calculate open returns
                open_returns = (current_open_data / prev_close_data - 1)

                # Identify stocks hitting upper and lower circuits
                upper_circuit_stocks = upper_circuit_tester(prev_close_data, current_open_data, open_returns, start_hold, filtered_stocks) 
                lower_circuit_stocks = lower_circuit_tester(prev_close_data, current_open_data, open_returns, start_hold, filtered_stocks)  

                # Update selected stocks to exclude those hitting upper circuit
                selected_stocks = upper_circuit_stocks[upper_circuit_stocks == False].index.tolist()

                # Get lists of stocks hitting circuits
                upper_circuit_stocks = [stock for stock in filtered_stocks if upper_circuit_stocks.get(stock, False)]
                lower_circuit_stocks = [stock for stock in filtered_stocks if lower_circuit_stocks.get(stock, False)]

                # Count circuit stocks
                num_upper_circuit_stocks = len(upper_circuit_stocks)
                num_lower_circuit_stocks = len(lower_circuit_stocks)

                # If no stocks are selected after filtering, return early with temp_result
                if not selected_stocks:
                    temp_result.update({
                        'num_upper_circuit_stocks': num_upper_circuit_stocks,
                        'upper_circuit_stocks': upper_circuit_stocks,
                        'num_lower_circuit_stocks': num_lower_circuit_stocks,
                        'lower_circuit_stocks': lower_circuit_stocks,
                    })
                    return temp_result
                
                # Calculate portfolio stock prices
                portfolio_stock_prices = start_prices[selected_stocks]

                # Calculate equated investment plan
                portfolio_info = calculate_equated_investment_plan(portfolio_stock_prices)
                stock_plan = portfolio_info['stock_plan']
                minimum_investment = portfolio_info['total_investment']

                # Calculate trading costs
                if selected_stocks == portfolio_info['num_shares_mapping'].index.tolist():
                    buy_val_per_stock = start_prices[selected_stocks] * portfolio_info['num_shares_mapping']
                    sell_val_per_stock = end_prices[selected_stocks] * portfolio_info['num_shares_mapping']
                else:
                    raise ValueError("Selected stocks missing in the investment plan.")

                trading_costs = calculate_trading_cost(buy_val_per_stock, sell_val_per_stock, hold_period) 

                # Calculate weighted portfolio return
                weighted_portfolio_return = calculate_weighted_portfolio_return(buy_val_per_stock, sell_val_per_stock, trading_costs, returns)
                portfolio_returns = weighted_portfolio_return['weighted_return']
                individual_returns = weighted_portfolio_return['individual_returns']

                # Calculate outperformance
                outperformance = portfolio_returns - avg_return if not np.isnan(portfolio_returns) else np.nan

                # Update result dictionary with hold period results
                result = {
                    'start_date': start_date,
                    'end_date': end_date,
                    'universe_return': avg_return,
                    'total_stocks': len(total_stocks),
                    'selected_stocks': selected_stocks,
                    'num_selected_stocks': len(selected_stocks),
                    'portfolio_return': portfolio_returns,
                    'outperformance': outperformance,
                    'weighted_portfolio_return': weighted_portfolio_return,
                    'total_minimum_investment_amount': minimum_investment,
                    'trading_costs': trading_costs,
                    'num_upper_circuit_stocks': num_upper_circuit_stocks,
                    'upper_circuit_stocks': upper_circuit_stocks,
                    'num_lower_circuit_stocks': num_lower_circuit_stocks,
                    'lower_circuit_stocks': lower_circuit_stocks,
                    '%_of_positive_stocks': individual_returns[individual_returns > 0].count() / len(individual_returns) if len(individual_returns) > 0 else 0,
                    '%_of_negative_stocks': individual_returns[individual_returns < 0].count() / len(individual_returns) if len(individual_returns) > 0 else 0,
                    'avg_positive_return': individual_returns[individual_returns > 0].mean() if individual_returns[individual_returns > 0].count() > 0 else np.nan,
                    'avg_negative_return': individual_returns[individual_returns < 0].mean() if individual_returns[individual_returns < 0].count() > 0 else np.nan,
                    'stock_price_ceiling': stock_price_ceiling, 
                }

        return result

    except KeyError as e:
        return {'error': f"Date {e} not found in price data."}
