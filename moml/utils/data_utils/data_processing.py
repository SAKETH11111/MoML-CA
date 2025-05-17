#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Common Data Processing Utilities

This module contains general-purpose data processing functions used across
different data processors in the MoML framework.
"""

import os
import pandas as pd
import numpy as np
import re
import logging
from pathlib import Path
from typing import Union, List, Dict, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("data_processing")


def load_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """Load data from a CSV file.

    Args:
        file_path: Path to the CSV file

    Returns:
        DataFrame containing the loaded data
    """
    logger.info(f"Loading data from: {file_path}")
    return pd.read_csv(file_path)


def inspect_data(df: pd.DataFrame) -> None:
    """Perform initial inspection of a dataset.

    Args:
        df: DataFrame to inspect
    """
    logger.info("=== Initial Data Inspection ===")
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Data types:\n{df.dtypes}")
    logger.info(f"Missing values:\n{df.isnull().sum()}")
    logger.info(f"First 5 rows:\n{df.head()}")


def clean_column_names(df: pd.DataFrame, column_mapping: Dict[str, str]) -> pd.DataFrame:
    """Rename columns for consistency.

    Args:
        df: DataFrame to clean
        column_mapping: Dictionary mapping old column names to new column names

    Returns:
        DataFrame with renamed columns
    """
    logger.info("=== Cleaning Column Names ===")
    df = df.rename(columns=column_mapping)
    logger.info("Columns renamed successfully")
    return df


def convert_numeric_columns(df: pd.DataFrame, numeric_columns: List[str]) -> pd.DataFrame:
    """Convert specified columns to numeric type.

    Args:
        df: DataFrame to convert
        numeric_columns: List of column names to convert to numeric

    Returns:
        DataFrame with converted numeric columns
    """
    logger.info("=== Converting Numeric Columns ===")

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            logger.info(f"Converted {col} to numeric")

    return df


def handle_missing_values(
    df: pd.DataFrame, numeric_fill: str = "median", categorical_fill: str = "Unknown"
) -> pd.DataFrame:
    """Handle missing values in the dataset.

    Args:
        df: DataFrame to process
        numeric_fill: Method to fill numeric missing values ('median' or 'mean')
        categorical_fill: Value to fill categorical missing values

    Returns:
        DataFrame with handled missing values
    """
    logger.info("=== Handling Missing Values ===")

    # Get numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns

    # Handle numeric columns
    for col in numeric_cols:
        if not df[col].isna().all():
            if numeric_fill == "median":
                fill_value = df[col].median()
            else:  # mean
                fill_value = df[col].mean()
            df[col] = df[col].fillna(fill_value)
            logger.info(f"Filled missing values in {col} with {numeric_fill}: {fill_value}")

    # Handle categorical columns
    for col in categorical_cols:
        df[col] = df[col].fillna(categorical_fill)
        logger.info(f"Filled missing values in {col} with '{categorical_fill}'")

    return df


def standardize_text_data(df: pd.DataFrame, text_columns: List[str], special_char_cols: List[str]) -> pd.DataFrame:
    """Standardize text data in specified columns.

    Args:
        df: DataFrame to standardize
        text_columns: List of column names containing text data
        special_char_cols: List of column names to remove special characters from

    Returns:
        DataFrame with standardized text data
    """
    logger.info("=== Standardizing Text Data ===")

    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].str.strip()
            logger.info(f"Removed trailing spaces from {col}")

            if col in special_char_cols:
                df[col] = df[col].str.replace(r"[^\w\s\-\(\)\[\]\{\}\.,;:+=/\\]", "", regex=True)
                logger.info(f"Removed special characters from {col}")

    return df


def extract_numeric_from_text(text: str) -> Optional[float]:
    """Extract numeric value from text string.

    Args:
        text: Text string containing numeric value

    Returns:
        Extracted numeric value or None if no value found
    """
    if pd.isna(text):
        return None

    text = str(text).lower()

    # Handle "ND" (Non-Detect) as 0
    if text == "nd" or text == "not detected":
        return 0.0

    # Extract first numeric value
    numeric_values = re.findall(r"\d+\.?\d*", text)
    return float(numeric_values[0]) if numeric_values else None


def save_processed_data(df: pd.DataFrame, output_path: Union[str, Path], create_dirs: bool = True) -> None:
    """Save processed DataFrame to CSV file.

    Args:
        df: DataFrame to save
        output_path: Path where to save the file
        create_dirs: Whether to create output directories if they don't exist
    """
    logger.info(f"=== Saving Processed Data to {output_path} ===")

    if create_dirs:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)
    logger.info("Data saved successfully")
