#%%

from datetime import datetime
import pandas as pd
from itertools import product
from multiprocessing import Pool
from database import create_database_engine, create_strategy_table, write_strategy_results, read_strategy_results
from utils import run_strategy_wrapper
import combinations_config

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
    all_combinations = combinations_config.combinations[0:5000]#list(product(test1_range, test2_range, hold_range, pct_selection, stock_price_ceiling))
    total_combinations = len(all_combinations)
    print(f"Total combinations to process: {total_combinations}")

    # Calculate dynamic batch size
    base_batch_size = 500  # Base size, divisible by 5 for 5 cores
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
    database = 'cmie_momentum'  # Update with your database name

    # Create database engine and table
    engine = create_database_engine(user, password, host, port, database)
    # create_strategy_table(engine)  # Uncomment if table needs to be created

    # Dictionary to store references to results (confirmation of storage)
    strategy_results = {}

    # Process combinations in batches
    for batch_start in range(0, len(all_combinations), batch_size):
        batch_combinations = all_combinations[batch_start:batch_start + batch_size]
        args_list = [(price_data_path, test1, test2, hold, pct, ceiling, begin, finish)
                     for test1, test2, hold, pct, ceiling in batch_combinations]

        # Use multiprocessing Pool with 5 processes to utilize 5 cores
        print(f"Processing batch {batch_start // batch_size + 1} of {(len(all_combinations) - 1) // batch_size + 1}...")

        try:
            with Pool(processes=5) as pool:  # Use 5 cores
                results = pool.map(run_strategy_wrapper, args_list)

            # Insert batch results into MySQL
            for strategy_name, results_df in results:
                if isinstance(results_df, pd.DataFrame) and not results_df.empty:
                    write_strategy_results(results_df, engine)
                    strategy_results[strategy_name] = "Stored in DB"
                    print(f"Completed strategy: {strategy_name}, stored in MySQL database")
                else:
                    print(f"Strategy {strategy_name} failed or returned no results: {results_df}")
        except Exception as e:
            print(f"Error processing batch {batch_start // batch_size + 1}: {str(e)}. Skipping this batch...")
            continue

    # Compile summary statistics
    summary_data = []

    for strategy_name in strategy_results.keys():
        # Query the database for this strategy's results
        results_df = read_strategy_results(strategy_name, engine)

        if results_df.empty:
            continue

        # Calculate outperformance
        portfolio_cum_return = results_df['portfolio_cum_return'].iloc[-1] - 1
        avg_cum_return = results_df['avg_cum_return'].iloc[-1] - 1
        outperformance = portfolio_cum_return - avg_cum_return

        # Calculate average minimum investment and trading cost
        avg_minimum_investment = results_df['total_minimum_investment_amount'].mean()

        summary_data.append({
            'Strategy_Name': strategy_name,
            'Portfolio_Returns': portfolio_cum_return,
            'Avg_Returns': avg_cum_return,
            'Outperformance': outperformance,
            'Strategy_Commence_Date': results_df['start_date'].iloc[0],
            'Strategy_End_Date': results_df['end_date'].iloc[-1],
            'Num_Processes': results_df['ProcessID'].max(),  # Number of periods processed
            'Avg_Portfolio_Return': results_df['portfolio_return'].mean(),
            'Avg_Outperformance': results_df['outperformance'].mean(),
            'Average_Minimum_Investment': avg_minimum_investment,
            'Outperformance_times_%' : (results_df['outperformance'] > 0).sum() / len(results_df)
        })

    # Create summary dataframe
    summary_df = pd.DataFrame(summary_data)

    # Save summary to Excel
    with pd.ExcelWriter(r'E:\Internship_Hashbrown\self\Relative Strength Momentum Strategy\Prototype\Results\momentum_strategy_results.xlsx') as writer:
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

    print("Analysis complete. Summary saved to momentum_strategy_results.xlsx")
    print("Detailed results are stored in the MySQL database table 'strategy_results'.")
    print("\nSummary Results:")
    print(summary_df)

    return {
        'strategy_results': strategy_results,  # Dictionary of storage confirmations
        'summary_results': summary_df         # Summary dataframe
    }

if __name__ == "__main__":
    price_data_path = r'E:\Internship_Hashbrown\self\Relative Strength Momentum Strategy\Source\CMIE_Price_Data_Cleaned.csv'
    begin = datetime(2014, 1, 1).date()
    finish = datetime(2024, 12, 31).date()

    # Run the multiple combinations test
    results = test_multiple_combinations(price_data_path, begin, finish)