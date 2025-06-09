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

