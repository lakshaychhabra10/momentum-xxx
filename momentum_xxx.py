#%%

import pandas as pd
import numpy as np
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
    
    stock_plan_df = pd.DataFrame.from_dict(stock_plan, orient='index')

    return {
        "total_investment": total_investment_required,
        "stock_plan": stock_plan_df,
        "stock_plan_str": stock_plan_df.to_string(),
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

    individual_returns = returns.loc[buy_val_per_stock.index].copy()

    individual_returns_str = ", ".join(
        f"{stock}: {returns[stock]:.4f}" for stock in buy_val_per_stock.index
    )

    return {
        "weighted_return": weighted_return,
        "individual_returns_series" : individual_returns,
        "individual_returns_str": individual_returns_str,
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
    june_2024 = pd.Timestamp('2024-06-01')
    threshold = 250 if open_date >= june_2024 else 100
    tick_low = 0.01
    tick_high = 0.05

    # Initialize result Series with False, only for the specified stocks
    result = pd.Series(False, index=close_price_data.index)

    # Function to round down to the nearest tick size
    def round_down_to_tick(price, tick_size):
        return np.floor(price / tick_size) * tick_size


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



def calculate_period_results(open_price_data, close_price_data,start_date, end_date,  pct_selection = 1,  eligible_stocks=None, hold=False, end_t2 = None , start_hold = None, hold_period = None, stock_price_ceiling = None, test1 = False, test2 = False):
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
        start_prices = open_price_data.loc[start_date]
        end_prices = close_price_data.loc[end_date]

        # Calculate returns
        # Align both start and end prices by index, then drop any stock with missing values
        combined_prices = pd.concat([start_prices, end_prices], axis=1, keys=['Open', 'Close']).dropna()
        # Calculate returns only on stocks with complete data
        returns = (combined_prices['Close'] / combined_prices['Open'] - 1)

        # Calculate average return and get total stocks
        avg_return = returns.mean()
        total_stocks = returns.index.tolist()

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
            'individual_returns_str': "",
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
            'stock_plan_str': "",
            }

            if not eligible_stocks:
                return temp_result

            if eligible_stocks:
                eligible_stocks = [stock for stock in eligible_stocks if stock in combined_prices.index]
                filtered_returns = returns[eligible_stocks]

                # Circuit testing
                prev_close_data = close_price_data.loc[end_t2]
                current_open_data = open_price_data.loc[start_hold]

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
                upper_circuit_stocks_bool = upper_circuit_tester(prev_close_data, current_open_data, open_returns, start_hold, filtered_stocks)

                # Update selected stocks to exclude those hitting upper circuit
                selected_stocks = upper_circuit_stocks_bool[upper_circuit_stocks_bool == False].index.tolist()

                # Get lists of stocks hitting circuits
                upper_circuit_stocks = upper_circuit_stocks_bool[upper_circuit_stocks_bool == True].index.tolist()

                # Count circuit stocks
                num_upper_circuit_stocks = len(upper_circuit_stocks)

                # If no stocks are selected after filtering, return early with temp_result
                if not selected_stocks:
                    temp_result.update({
                        'num_upper_circuit_stocks': num_upper_circuit_stocks,
                        'upper_circuit_stocks': upper_circuit_stocks,
                    })
                    return temp_result

                # Calculate portfolio stock prices
                portfolio_stock_prices = start_prices[selected_stocks]

                # Calculate equated investment plan
                portfolio_info = calculate_equated_investment_plan(portfolio_stock_prices)
                stock_plan_str = portfolio_info['stock_plan_str']
                minimum_investment = portfolio_info['total_investment']

                # Calculate trading costs
                buy_val_per_stock = start_prices[selected_stocks] * portfolio_info['num_shares_mapping']
                sell_val_per_stock = end_prices[selected_stocks] * portfolio_info['num_shares_mapping']


                trading_costs = calculate_trading_cost(buy_val_per_stock, sell_val_per_stock, hold_period)

                # Calculate weighted portfolio return
                weighted_portfolio_return = calculate_weighted_portfolio_return(buy_val_per_stock, sell_val_per_stock, trading_costs, returns)
                portfolio_returns = weighted_portfolio_return['weighted_return']
                individual_returns_str = weighted_portfolio_return['individual_returns_str']
                individual_returns = weighted_portfolio_return['individual_returns_series']

                # Calculate outperformance
                outperformance = portfolio_returns - avg_return if not np.isnan(portfolio_returns) else None
                pos_returns = individual_returns[individual_returns > 0]
                neg_returns = individual_returns[individual_returns < 0]

                # Update result dictionary with hold period results
                result = {
                    'start_date': start_date,
                    'end_date': end_date,
                    'universe_return': avg_return,
                    'total_stocks': len(total_stocks),
                    'selected_stocks': selected_stocks,
                    'individual_returns_str': individual_returns_str,
                    'num_selected_stocks': len(selected_stocks),
                    'portfolio_return': portfolio_returns,
                    'outperformance': outperformance,
                    'weighted_portfolio_return': weighted_portfolio_return,
                    'total_minimum_investment_amount': minimum_investment,
                    'trading_costs': trading_costs,
                    'num_upper_circuit_stocks': num_upper_circuit_stocks,
                    'upper_circuit_stocks': upper_circuit_stocks,
                    '%_of_positive_stocks': pos_returns.count() / len(individual_returns) if len(individual_returns) > 0 else 0,
                    '%_of_negative_stocks': neg_returns.count() / len(individual_returns) if len(individual_returns) > 0 else 0,
                    'avg_positive_return': pos_returns.mean() if pos_returns.count() > 0 else None,
                    'avg_negative_return': neg_returns.mean() if neg_returns.count() > 0 else None,
                    'stock_price_ceiling': stock_price_ceiling,
                    'stock_plan_str': stock_plan_str,
                }

        return result

    except KeyError as e:
        return {'error': f"Date {e} not found in price data."}

"""## Strategy"""

import numpy as np
import pandas as pd
from datetime import datetime
import math

def run_momentum_strategy(open_price_data, close_price_data,test1_period, test2_period, hold_period, begin, finish, pct_selection=0.5, stock_price_ceiling=None):
    """
    Runs the momentum strategy for relative strength momentum trading.

    Args:
        price_data (csv): price data CSV file.
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

    # Sort and filter data
    open_price_data = open_price_data[(open_price_data.index >= begin) & (open_price_data.index <= finish)]
    close_price_data = close_price_data[(close_price_data.index >= begin) & (close_price_data.index <= finish)]

    # Ensure both have same dates
    if not open_price_data.index.equals(close_price_data.index):
        raise ValueError("Date indices of open and close price data do not match after filtering.")

    available_dates = open_price_data.index.tolist()

    # Check if there are enough dates
    min_required_dates = (test1_period + test2_period + hold_period)
    if len(available_dates) < min_required_dates:
        raise ValueError("Not enough dates to run the strategy.")

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
            open_price_data=open_price_data,
            close_price_data=close_price_data,
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
            open_price_data=open_price_data,
            close_price_data=close_price_data,
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
            open_price_data=open_price_data,
            close_price_data=close_price_data,
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
            'portfolio_stocks': ','.join(hold_results['selected_stocks']),
            'individual_returns_str': hold_results['individual_returns_str'],
            'start_date': hold_results['start_date'],
            'end_date': hold_results['end_date'],
            'universe_return': round(hold_results['universe_return'], 4),
            'portfolio_return': round(hold_results['portfolio_return'],4),
            'outperformance': round(hold_results['outperformance'],4),
#            'total_stocks': hold_results['total_stocks'],
            'total_minimum_investment_amount': round(hold_results['total_minimum_investment_amount'],4), #hold_results['total_minimum_investment_amount'],
#            'trading_costs': hold_results['trading_costs'],
#            'num_upper_circuit_stocks': hold_results['num_upper_circuit_stocks'],
#            'num_lower_circuit_stocks': hold_results['num_lower_circuit_stocks'],
            '%_of_positive_stocks': hold_results['%_of_positive_stocks'],
            '%_of_negative_stocks': hold_results['%_of_negative_stocks'],
            'avg_positive_return': hold_results['avg_positive_return'],
            'avg_negative_return': hold_results['avg_negative_return'],
            'upper_circuit_stocks': ','.join(hold_results['upper_circuit_stocks']),
#            'lower_circuit_stocks': ','.join(hold_results['lower_circuit_stocks']),
            'stock_price_ceiling': stock_price_ceiling,
            'stock_plan_str': hold_results['stock_plan_str'],
            'total_stocks': hold_results['total_stocks']
        }

        # Update index and ProcessID
        start_idx = start_idx + hold_period
        ProcessID += 1

        # Append result to DataFrame
        result_df = pd.concat([result_df, pd.DataFrame([result_dict])], ignore_index=True)

    # Appending columns in DataFrame
    if not result_df.empty:
        result_df['portfolio_cum_return'] = (1 + result_df['portfolio_return']).cumprod()
        result_df['universe_cum_return'] = (1 + result_df['universe_return']).cumprod()
        result = result_df

        # Calculate consecutive overlap
#        result = calculate_consecutive_overlap(result_df, stocks_col='stocks', max_back_steps=5, drop_temp=True)

    # Save and print results
    if not result.empty:
        pass
    else:
        print("Strategy failed due to insufficient data or errors.")

    return result


"""## Database"""

from sqlalchemy import create_engine


def create_database_engine(username, password, host , database):
    """
    Creates a SQLAlchemy engine for connecting to the MySQL database.

    Args:
        user (str): MySQL username.
        password (str): MySQL password.
        host (str): MySQL host address.
        database (str): MySQL database name.

    Returns:
        sqlalchemy.engine.Engine: SQLAlchemy engine for database connection.
    """
    return create_engine(f"mysql+pymysql://{username}:{password}@{host}/{database}", pool_pre_ping=True)

# Database connection configuration
username = 'lakshay'  # Update with your MySQL username
password = 'Lakshay12'  # Update with your MySQL password
host = '34.126.222.132'
database = 'momentum'  # Update with your database name

# Create database engine and table
engine = create_database_engine(username, password, host, database)

def write_strategy_results(results_df, engine):
    """
    Writes strategy results to the MySQL database.

    Args:
        results_df (pd.DataFrame): DataFrame containing strategy results.
        engine (sqlalchemy.engine.Engine): SQLAlchemy engine for database connection.
    """
    results_df.to_sql(
        name='strategy_results',
        con=engine,
        if_exists='append',  # Append to existing table
        index=False  # Don't write DataFrame index as a column
    )


"""## Combinations_config"""

from itertools import product
import numpy as np

test1_range = [10]
test2_range = [85]
hold_range = [95]
pct_selection = [0.4]
stock_price_ceiling = [1800]


# test1_range = list(range(2, 90, 3))  # Test1 period range
# test2_range = list(range(2, 253, 3))  # Test1 period range
# hold_range = list(range(2, 253, 3))  # Test1 period range

# pct_selection = np.arange(0.1, 1.0, 0.1).tolist()  # Converts to Python list
# stock_price_ceiling = [5000,500000]  # Price ceiling for stock selection


# Generate all combinations
#combinations = list(product(test1_range, test2_range, hold_range, pct_selection, stock_price_ceiling))
#len(combinations)
# combinations = [(10,10,10,1,1200)]

combinations = [
    (35, 85, 55, 0.1, 500000)
]


"""## Wrapper"""

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
    open_price_data, close_price_data ,test1, test2, hold, pct_selection, stock_price_ceiling, begin, finish = args
    strategy_name = f"{test1}-{test2}-{hold}-pct{pct_selection}-ceiling{stock_price_ceiling}"
    results_df = run_momentum_strategy(
        open_price_data=open_price_data,
        close_price_data=close_price_data,
        test1_period=test1,
        test2_period=test2,
        hold_period=hold,
        begin=begin,
        finish=finish,
        pct_selection=pct_selection,
        stock_price_ceiling=stock_price_ceiling
    )
    return strategy_name, results_df

"""## CPU Main"""

from datetime import datetime
import pandas as pd
from itertools import product
from multiprocessing import Pool
import logging
from google.cloud import storage
from datetime import datetime
from google.api_core.exceptions import NotFound

def setup_logger():
    """Configure logging to both file and console"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('momentum_strategy.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()


def gcs_file_exists(bucket_name, blob_name):
    """Check if a file exists in GCS"""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        return blob.exists()
    except Exception as e:
        logging.error(f"GCS existence check failed for {blob_name}: {str(e)}")
        return False


def upload_to_gcs(bucket_name, source_file, destination_blob_name):
    """Uploads a file to Google Cloud Storage"""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file)
        return True
    except Exception as e:
        logging.error(f"Failed to upload {source_file} to GCS: {str(e)}")
        return False

def test_multiple_combinations(open_price_data_path, close_price_data_path , begin, finish, gcs_bucket=None):
    """
    Tests multiple combinations of parameters using momentum strategy.
    Saves each batch as CSV and optionally uploads to Google Cloud Storage.
    """
    logger = setup_logger()
    
    # Load and format price data
    try:
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

        # open_price_data.to_csv('open_review.csv')
        # close_price_data.to_csv('close_review.csv')

    except Exception as e:
        logger.error(f"Error loading price data: {str(e)}")
        return

    # Generate all combinations
    all_combinations = combinations  # list(product(...))

    batch_size = max(100, 5)
    custom_break_batch = None
    start_from_batch = 1

    logger.info(f"Starting processing... Total Combinations : {len(all_combinations)} , Total Batches : {math.ceil(len(all_combinations) / batch_size)} , Batch Size : {batch_size}, Break Batch at : Batch {custom_break_batch}")

    for batch_num, batch_start in enumerate(range(0, len(all_combinations), batch_size), 1):
    
        if batch_num < start_from_batch:
            continue

        if custom_break_batch is not None and batch_num > custom_break_batch:
            logger.info(f"Stopping script after batch {custom_break_batch} as per custom breakpoint.")
            break

        csv_filename = f"batch_{batch_num}.csv"
        gcs_blob_path = f"strategy_results/{csv_filename}"

        # Skip if batch result already exists
        # if gcs_bucket and gcs_file_exists(gcs_bucket, gcs_blob_path):
        #     logger.info(f"Skipping batch {batch_num} as {gcs_blob_path} already exists on GCS.")
        #     continue

        batch_combinations = all_combinations[batch_start:batch_start + batch_size]
        args_list = [(open_price_data, close_price_data,test1, test2, hold, pct, ceiling, begin, finish)
                    for test1, test2, hold, pct, ceiling in batch_combinations]


        try:
            with Pool(processes=5) as pool:
                results = list(pool.imap(run_strategy_wrapper, args_list))

            # Process valid results
            valid_results = [(name, df) for name, df in results if isinstance(df, pd.DataFrame) and not df.empty]
            
            if valid_results:
                try:
                    combined_df = pd.concat([df for _, df in valid_results], ignore_index=True)
                    write_strategy_results(combined_df, engine)
                except Exception as e:
                    logger.error(f"Error storing strategy results in batch {batch_num}: {str(e)}")

                # Generate batch summary
                batch_summary = []
                for strategy_name, df in valid_results:
                    logger.info(f"Successfully processed {strategy_name} in batch {batch_num}")

                    portfolio_cum_return = df['portfolio_cum_return'].iloc[-1] - 1
                    avg_cum_return = df['universe_cum_return'].iloc[-1] - 1
                    outperformance = portfolio_cum_return - avg_cum_return
                    avg_minimum_investment = df['total_minimum_investment_amount'].mean()

                    batch_summary.append({
                        'Strategy_Name': strategy_name,
                        'Portfolio_Returns': round(portfolio_cum_return, 4),
                        'Universe_Returns': round(avg_cum_return, 4),
                        'Outperformance': round(outperformance, 4),
                        'Strategy_Commence_Date': df['start_date'].iloc[0],
                        'Strategy_End_Date': df['end_date'].iloc[-1],
                        'Num_Processes': df['ProcessID'].max(),
                        'Avg_Portfolio_Return': round(df['portfolio_return'].mean(), 4),
                        'Avg_Outperformance': round(df['outperformance'].mean(), 4),
                        'Average_Minimum_Investment': int(avg_minimum_investment),
                        '%_periods_outperformed': round((df['outperformance'] > 0).sum() / len(df), 4)
                    })

                # Save batch CSV locally
                try:
                    batch_df = pd.DataFrame(batch_summary)
                    csv_filename = f"batch_{batch_num}.csv"
                    batch_df.to_csv(csv_filename, index=False)
                except Exception as e:
                    logger.error(f"Error saving batch {batch_num} to CSV: {str(e)}")

                # Upload to GCS if bucket specified
                if gcs_bucket:
                    gcs_path = f"strategy_results/{csv_filename}"
                    upload_to_gcs(gcs_bucket, csv_filename, gcs_path)
                    # No success log, function logs error if fail

            else:
                logger.warning(f"No valid strategy results in batch {batch_num}")

            # Log failures
            failed_strategies = [name for name, df in results if not isinstance(df, pd.DataFrame) or df.empty]
            if failed_strategies:
                logger.warning(f"Failed or empty strategies in batch {batch_num}: {failed_strategies}")

        except Exception as e:
            logger.error(f"Error in batch {batch_num}: {str(e)}", exc_info=True)
            continue
            
if __name__ == "__main__":
    open_price_data_path = r'price_data/open_data.csv'
    close_price_data_path = r'price_data/close_data.csv'
    begin = pd.Timestamp('2014-01-01')
    finish = pd.Timestamp('2024-12-31')

    test_multiple_combinations(open_price_data_path=open_price_data_path, close_price_data_path=close_price_data_path, begin=begin, finish=finish, gcs_bucket=None)
