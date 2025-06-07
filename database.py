# -*- coding: utf-8 -*-
"""
Database interactions for storing and retrieving momentum strategy results.
"""
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

def create_database_engine(user, password, host, port , database):
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
    return create_engine(f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}")

def create_strategy_table(engine):
    """
    Creates the strategy_results table in the MySQL database if it doesn't exist.

    Args:
        engine (sqlalchemy.engine.Engine): SQLAlchemy engine for database connection.
    """
    create_table_query = """
    CREATE TABLE IF NOT EXISTS strategy_results (
        id INT AUTO_INCREMENT PRIMARY KEY,
        strategyID VARCHAR(20),
        ProcessID INT,
        test_1_stocks INT,
        test_2_stocks INT,
        stocks TEXT,
        start_date DATE,
        end_date DATE,
        avg_return FLOAT,
        portfolio_return FLOAT,
        outperformance FLOAT,
        total_stocks INT,
        portfolio_cum_return FLOAT,
        avg_cum_return FLOAT
    )
    """
    with engine.connect() as conn:
        conn.execute(text(create_table_query))

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

def read_strategy_results(strategy_name, engine):
    """
    Reads strategy results from the MySQL database for a given strategy.

    Args:
        strategy_name (str): Name of the strategy (e.g., "30-30-30-pct0.3-ceiling2000").
        engine (sqlalchemy.engine.Engine): SQLAlchemy engine for database connection.

    Returns:
        pd.DataFrame: DataFrame containing results for the specified strategy.
    """
    try:
        # Parse the strategy_name into components
        # Expected format: "test1-test2-hold-pctX-ceilingY"
        parts = strategy_name.split('-')
        if len(parts) != 5 or parts[3].startswith('pct') is False or parts[4].startswith('ceiling') is False:
            raise ValueError(f"Invalid strategy_name format: {strategy_name}. Expected format: test1-test2-hold-pctX-ceilingY")

        # Extract strategyID (e.g., "30-30-30")
        strategy_id = f"{parts[0]}-{parts[1]}-{parts[2]}"

        # Extract pct_selection (e.g., "0.3" from "pct0.3")
        pct_selection = float(parts[3].replace('pct', ''))

        # Extract stock_price_ceiling (e.g., "2000" from "ceiling2000")
        stock_price_ceiling = float(parts[4].replace('ceiling', ''))

        # Construct the query to match strategyID, pct_selection, and stock_price_ceiling
        query = """
            SELECT * FROM strategy_results
            WHERE strategyID = %s
            AND pct_selection = %s
            AND stock_price_ceiling = %s
        """
        params = (strategy_id, pct_selection, stock_price_ceiling)

        # Execute the query
        df = pd.read_sql(query, engine, params=params)

        if df.empty:
            print(f"No results found for strategy: {strategy_name}")
        
        return df

    except ValueError as e:
        print(f"Error parsing strategy_name: {str(e)}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error querying database for strategy {strategy_name}: {str(e)}")
        return pd.DataFrame()