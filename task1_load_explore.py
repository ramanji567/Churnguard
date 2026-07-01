"""
ChurnGuard - Task 1: Load & Explore the Dataset
--------------------------------------------------
Loads the raw churnguard_data.csv and produces a clear summary of its
structure and data quality issues (missing values, duplicates,
inconsistent categories, typos) before any cleaning is done.
"""

import pandas as pd

# 1. Load the dataset
df = pd.read_csv("churnguard_data.csv")

# 2. Shape of the dataset
print("=" * 60)
print("SHAPE OF DATASET (rows, columns)")
print("=" * 60)
print(df.shape)

# 3. First 5 rows
print("\n" + "=" * 60)
print("FIRST 5 ROWS")
print("=" * 60)
print(df.head())

# 4. Column names and data types
print("\n" + "=" * 60)
print("COLUMN INFO (.info())")
print("=" * 60)
df.info()

# 5. Missing values per column
print("\n" + "=" * 60)
print("MISSING VALUES PER COLUMN")
print("=" * 60)
print(df.isnull().sum())

# 6. Duplicate rows
print("\n" + "=" * 60)
print("NUMBER OF DUPLICATE ROWS")
print("=" * 60)
print(df.duplicated().sum())

# 7. Value counts of Churn (inconsistent entries expected)
print("\n" + "=" * 60)
print("VALUE COUNTS - Churn column")
print("=" * 60)
print(df["Churn"].value_counts(dropna=False))

# 8. Unique values in Contract (typos expected)
print("\n" + "=" * 60)
print("UNIQUE VALUES - Contract column")
print("=" * 60)
print(df["Contract"].unique())
