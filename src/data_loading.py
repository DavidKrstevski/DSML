"""Helpers for loading and inspecting the dataset."""

import pandas as pd

# Plotting style is configured in the visualization module

def load_dataset(path='sleep_cycle_productivity.csv'):
    """Load the dataset from *path* and perform sanity checks."""
    # Read the CSV file into a DataFrame
    df = pd.read_csv(path)

    #Inspect : shape, types, preview
    print("=== First 5 rows of the data ===")
    print(df.head())

    # Basic frame information
    print("\nDataframe shape:", df.shape)
    print("\nData types:\n", df.dtypes)

    # Look for missing values
    print("\nmissing values per column:")
    print(df.isnull().sum())

    # Check for duplicates
    duplicates = df.duplicated().sum()
    print(f"\nnumber of duplicate rows: {duplicates}")

    # Quick summary statistics for a sanity check
    print("\nsummary statistics (numerical columns):")
    print(df.describe())

    # Example: convert strings to numbers if necessary
    # df['column_name'] = pd.to_numeric(df['column_name'], errors='coerce')

    # Value counts for categorical features (if any)
    cat_columns = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_columns:
        print(f"\nvalue counts for {col}:")
        print(df[col].value_counts())

    # Correlation matrix for numerical features
    print("\ncorrelation matrix:")
    corr_matrix = df.corr(numeric_only=True)
    print(corr_matrix)

    return df, corr_matrix
