"""
ChurnGuard - Task 2: Clean the Dataset
--------------------------------------------------
Applies a repeatable, step-by-step cleaning pipeline to the raw
churnguard_data.csv and produces a ready-to-use DataFrame.
"""

import pandas as pd

# 1. Load the dataset
df = pd.read_csv("churnguard_data.csv")

# 2. Drop the customerID column (not useful for modelling)
df = df.drop(columns=["customerID"])

# 3. Remove duplicate rows
df = df.drop_duplicates()

# 4. Strip whitespace from gender and PaymentMethod
df["gender"] = df["gender"].str.strip()
df["PaymentMethod"] = df["PaymentMethod"].str.strip()

# 5. Standardise casing -> title case for Churn, PhoneService, PaperlessBilling
for col in ["Churn", "PhoneService", "PaperlessBilling"]:
    df[col] = df[col].str.strip().str.title()

# 6. Fix Contract -> map all variations to one of three valid values
contract_map = {
    "month-to-month": "Month-to-month",
    "month to month": "Month-to-month",
    "monthly": "Month-to-month",
    "one year": "One year",
    "1 year": "One year",
    "two year": "Two year",
    "2 year": "Two year",
}
df["Contract"] = (
    df["Contract"].str.strip().str.lower().map(contract_map).fillna(df["Contract"])
)

# 7. Fix InternetService -> map all variations to one of three valid values
internet_map = {
    "dsl": "DSL",
    "fiber optic": "Fiber optic",
    "fibre optic": "Fiber optic",
    "fiberoptic": "Fiber optic",
    "no": "No",
    "none": "No",
}
df["InternetService"] = (
    df["InternetService"]
    .str.strip()
    .str.lower()
    .map(internet_map)
    .fillna(df["InternetService"])
)

# 8. Fix TotalCharges -> convert to numeric, junk values become NaN
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# 9. Remove rows where tenure is zero or negative
df = df[~(df["tenure"] <= 0)]

# 10. Remove rows where MonthlyCharges is less than 10 or greater than 200
df = df[~((df["MonthlyCharges"] < 10) | (df["MonthlyCharges"] > 200))]

# 11. Fill missing values
df["MonthlyCharges"] = df["MonthlyCharges"].fillna(df["MonthlyCharges"].mean())
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].mean())
df["tenure"] = df["tenure"].fillna(round(df["tenure"].median())).astype(int)

# 12. Print the shape of the cleaned DataFrame
print("=" * 60)
print("SHAPE OF CLEANED DATASET (rows, columns)")
print("=" * 60)
print(df.shape)

# 13. Print missing value counts to confirm all issues are resolved
print("\n" + "=" * 60)
print("MISSING VALUE COUNTS AFTER CLEANING")
print("=" * 60)
print(df.isnull().sum())
