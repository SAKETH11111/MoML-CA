"""
moml/utils/data_utils/data_processing.py

This module provides a collection of general-purpose utility functions for
common data cleaning and preprocessing tasks using the pandas library. These
functions are designed to be modular and reusable across different data
pipelines within the MoML-CA framework.
"""

import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

# Logger for this module
logger = logging.getLogger(__name__)


def load_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load data from a CSV file into a pandas DataFrame.

    Args:
        file_path (Union[str, Path]): The path to the CSV file.

    Returns:
        pd.DataFrame: A DataFrame containing the loaded data.
    """
    logger.info(f"Loading data from: {file_path}")
    return pd.read_csv(file_path)


def inspect_data(df: pd.DataFrame) -> None:
    """
    Perform and log a basic inspection of a DataFrame.

    This function provides a quick overview of the DataFrame's properties,
    including its shape, data types, and the extent of missing values.

    Args:
        df (pd.DataFrame): The DataFrame to inspect.
    """
    logger.info("--- Initial Data Inspection ---")
    logger.info(f"Dataset Shape: {df.shape}")
    logger.info(f"Data Types:\n{df.dtypes.to_string()}")
    logger.info(f"Missing Values per Column:\n{df.isnull().sum().to_string()}")
    logger.info("---------------------------------")


def clean_column_names(df: pd.DataFrame, column_mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Rename DataFrame columns for clarity and consistency.

    Args:
        df (pd.DataFrame): The DataFrame whose columns will be renamed.
        column_mapping (Dict[str, str]): A dictionary mapping original column
            names to their new names.

    Returns:
        pd.DataFrame: A new DataFrame with the renamed columns.
    """
    logger.info("Cleaning column names...")
    df = df.rename(columns=column_mapping)
    logger.info("Column names cleaned successfully.")
    return df


def convert_numeric_columns(df: pd.DataFrame, numeric_columns: List[str]) -> pd.DataFrame:
    """
    Convert specified columns of a DataFrame to a numeric type.

    Non-convertible values will be set to NaN (Not a Number).

    Args:
        df (pd.DataFrame): The DataFrame to process.
        numeric_columns (List[str]): A list of column names to convert.

    Returns:
        pd.DataFrame: The DataFrame with specified columns converted to numeric types.
    """
    logger.info("Converting columns to numeric type...")
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            logger.info(f"  - Converted '{col}' to numeric.")
    return df


def handle_missing_values(
    df: pd.DataFrame, numeric_fill: str = "median", categorical_fill: str = "Unknown"
) -> pd.DataFrame:
    """
    Impute missing values in a DataFrame.

    Fills missing values in numeric columns with the column's median or mean,
    and fills missing values in categorical columns with a specified string.

    Args:
        df (pd.DataFrame): The DataFrame with missing values to handle.
        numeric_fill (str, optional): The method for filling numeric columns,
            either 'median' or 'mean'. Defaults to "median".
        categorical_fill (str, optional): The placeholder value for missing
            categorical data. Defaults to "Unknown".

    Returns:
        pd.DataFrame: The DataFrame with missing values imputed.
    """
    logger.info("Handling missing values...")
    numeric_cols = df.select_dtypes(include=np.number).columns
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns

    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            fill_value = df[col].median() if numeric_fill == "median" else df[col].mean()
            df[col].fillna(fill_value, inplace=True)
            logger.info(f"  - Filled missing values in numeric column '{col}' with {numeric_fill} ({fill_value:.4f}).")

    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(categorical_fill, inplace=True)
            logger.info(f"  - Filled missing values in categorical column '{col}' with '{categorical_fill}'.")

    return df


def standardize_text_data(df: pd.DataFrame, text_columns: List[str]) -> pd.DataFrame:
    """
    Standardize text data by stripping whitespace.

    Args:
        df (pd.DataFrame): The DataFrame to process.
        text_columns (List[str]): A list of column names containing text data.

    Returns:
        pd.DataFrame: The DataFrame with standardized text data.
    """
    logger.info("Standardizing text data...")
    for col in text_columns:
        if col in df.columns and df[col].dtype == "object":
            df[col] = df[col].str.strip()
            logger.info(f"  - Stripped whitespace from '{col}'.")
    return df


def extract_numeric_from_text(text: str) -> Optional[float]:
    """
    Extract the first numeric value found in a text string.

    This function also recognizes "ND" or "not detected" as zero.

    Args:
        text (str): The text string to parse.

    Returns:
        Optional[float]: The extracted numeric value, or None if no numeric
        value is found.
    """
    if pd.isna(text):
        return None

    text_lower = str(text).lower()
    if text_lower in ["nd", "not detected"]:
        return 0.0

    # This regex finds the first integer or floating-point number.
    match = re.search(r"\d+\.?\d*", text_lower)
    return float(match.group(0)) if match else None


def save_processed_data(df: pd.DataFrame, output_path: Union[str, Path]) -> None:
    """
    Save a processed DataFrame to a CSV file.

    This function ensures the output directory exists before saving.

    Args:
        df (pd.DataFrame): The DataFrame to be saved.
        output_path (Union[str, Path]): The destination file path.
    """
    logger.info(f"Saving processed data to: {output_path}")
    output_dir = Path(output_path).parent
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Data saved successfully.")
