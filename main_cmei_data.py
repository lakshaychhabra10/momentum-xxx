#%%

from datetime import datetime
import pandas as pd
from itertools import product
from multiprocessing import Pool
from database import create_database_engine, write_strategy_results
from utils import run_strategy_wrapper
import combinations_config
import cProfile
import pstats
import os
import time
from tqdm import tqdm
import sys

def test_multiple_combinations(price_data_path, begin=datetime(2019, 3, 28).date(), finish=datetime(2025, 3, 28).date()):
    """
    Tests multiple combinations of test1_period, test2_period, hold_period, pct_selection, and stock_price_ceiling
    using the momentum strategy. Processes combinations in parallel, stores results in MySQL, and compiles a summary.

    Args:
        price_data_path (str): Path to the CSV file containing price data.
        begin (datetime.date): Start date for the strategy.
        finish (datetime.date): End date for the strategy.

    Returns:
        dict: Dictionary with 'strategy_results' (storage confirmations) and 'summary_results' (summary DataFrame).
    """

    # Generate all combinations
    all_combinations = combinations_config.combinations[0:5]#list(product(test1_range, test2_range, hold_range, pct_selection, stock_price_ceiling))
    total_combinations = len(all_combinations)
    print(f"Total combinations to process: {total_combinations}")

    # Calculate dynamic batch size
    base_batch_size = 10  # Base size, divisible by 5 for 5 cores
    threshold = 10000  # Threshold for scaling batch size
    max_batch_size = 5000  # Cap to keep memory usage under 500 MB

    if total_combinations <= threshold:
        batch_size = base_batch_size
    else:
        # Scale batch size proportionally, but cap at max_batch_size
        scaling_factor = total_combinations / threshold
        batch_size = int(base_batch_size * scaling_factor)
        batch_size = min(batch_size, max_batch_size)
        # Ensure batch_size is divisible by 5 for even distribution across 5 cores
        batch_size = (batch_size // 5) * 5

    batch_size = max(batch_size, 5)  # Ensure at least 6 to utilize all cores
    print(f"Dynamic batch size set to: {batch_size}")

    # Database connection configuration
    user = 'root'  # Update with your MySQL username
    password = 'Lakshay%4012'  # Update with your MySQL password
    port = 3306
    host = 'localhost'
    database = 'test'  # Update with your database name

    # Create database engine and table
    engine = create_database_engine(user, password, host, port, database)
    # create_strategy_table(engine)  # Uncomment if table needs to be created


    summary_data = []  # Final summary for all batches

    for batch_start in tqdm(range(0, len(all_combinations), batch_size), desc="Processing batches"):
        batch_combinations = all_combinations[batch_start:batch_start + batch_size]
        args_list = [(price_data_path, test1, test2, hold, pct, ceiling, begin, finish)
                    for test1, test2, hold, pct, ceiling in batch_combinations]

        print(f"Processing batch {batch_start // batch_size + 1} of {(len(all_combinations) - 1) // batch_size + 1}...")

        try:
            with Pool(processes=5) as pool:
                results = list(tqdm(pool.imap(run_strategy_wrapper, args_list), total=len(args_list), desc="Running strategies"))

            valid_results = [(name, df) for name, df in results if isinstance(df, pd.DataFrame) and not df.empty]

            if valid_results:
                combined_df = pd.concat([df for _, df in valid_results], ignore_index=True)
                write_strategy_results(combined_df, engine)
                print(f"✅ Stored {len(valid_results)} strategy results in batch {batch_start // batch_size + 1}.")

                # ✅ Create summary just for this batch
                for strategy_name, df in valid_results:
                    if df.empty:
                        continue
                    portfolio_cum_return = df['portfolio_cum_return'].iloc[-1] - 1
                    avg_cum_return = df['universe_cum_return'].iloc[-1] - 1
                    outperformance = portfolio_cum_return - avg_cum_return
                    avg_minimum_investment = df['total_minimum_investment_amount'].mean()

                    summary_data.append({
                        'Strategy_Name': strategy_name,
                        'Portfolio_Returns': round(portfolio_cum_return, 2),
                        'Universe_Returns': round(avg_cum_return, 2),
                        'Outperformance': round(outperformance, 2),
                        'Strategy_Commence_Date': df['start_date'].iloc[0],
                        'Strategy_End_Date': df['end_date'].iloc[-1],
                        'Num_Processes': df['ProcessID'].max(),
                        'Avg_Portfolio_Return': round(df['portfolio_return'].mean(), 4),
                        'Avg_Outperformance': round(df['outperformance'].mean(), 4),
                        'Average_Minimum_Investment': int(avg_minimum_investment),
                        'Outperformance_times_%': round((df['outperformance'] > 0).sum() / len(df), 2)
                    })
            else:
                print("⚠️ No valid strategy results to store in this batch.")

            failed_strategies = [name for name, df in results if not isinstance(df, pd.DataFrame) or df.empty]
            if failed_strategies:
                print(f"❌ Failed or empty: {failed_strategies}")

        except Exception as e:
            print(f"💥 Error in batch {batch_start // batch_size + 1}: {str(e)}. Skipping this batch.")
            continue

    # ✅ After all batches
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_excel('momentum_strategy_results.xlsx', index=False)
    print("📦 Final summary saved to momentum_strategy_results.xlsx")

if __name__ == "__main__":
    price_data_path = r'cmie\CMIE_Price_Data_Cleaned.csv'
    begin = datetime(2014, 1, 1).date()
    finish = datetime(2024, 12, 31).date()

    test_multiple_combinations(price_data_path, begin, finish)
