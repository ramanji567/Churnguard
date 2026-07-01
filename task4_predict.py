"""
ChurnGuard - Task 4: Build a Prediction Script
--------------------------------------------------
Retrains a Logistic Regression model on the FULL cleaned dataset using
five numeric/ordinal features, then collects a specific customer's
details from the user and prints an instant churn prediction.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression


def load_and_clean(path="churnguard_data.csv"):
    """Apply all Task 2 cleaning steps and return a ready-to-use DataFrame."""
    df = pd.read_csv(path)

    df = df.drop(columns=["customerID"])
    df = df.drop_duplicates()

    df["gender"] = df["gender"].str.strip()
    df["PaymentMethod"] = df["PaymentMethod"].str.strip()

    for col in ["Churn", "PhoneService", "PaperlessBilling"]:
        df[col] = df[col].str.strip().str.title()

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

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    df = df[~(df["tenure"] <= 0)]
    df = df[~((df["MonthlyCharges"] < 10) | (df["MonthlyCharges"] > 200))]

    df["MonthlyCharges"] = df["MonthlyCharges"].fillna(df["MonthlyCharges"].mean())
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].mean())
    df["tenure"] = df["tenure"].fillna(round(df["tenure"].median())).astype(int)

    return df


# 1. Load and clean the dataset
df = load_and_clean("churnguard_data.csv")

# 2. Encode Churn: Yes -> 1, No -> 0
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Contract needs a numeric/ordinal encoding for the manual-input model
# 0 = Month-to-month, 1 = One year, 2 = Two year
contract_order = {"Month-to-month": 0, "One year": 1, "Two year": 2}
df["Contract"] = df["Contract"].map(contract_order)

# Feature set for this task: five numeric/ordinal features only
FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen", "Contract"]

model_df = df.dropna(subset=FEATURES + ["Churn"])
X = model_df[FEATURES]
y = model_df["Churn"]

# 3. Retrain Logistic Regression on the FULL cleaned dataset (no split)
model = LogisticRegression(max_iter=1000)
model.fit(X, y)


def get_int_input(prompt, valid_values=None):
    while True:
        raw = input(prompt).strip()
        try:
            value = int(raw)
            if valid_values is not None and value not in valid_values:
                print(f"Please enter one of {valid_values}.")
                continue
            return value
        except ValueError:
            print("Please enter a whole number.")


def get_float_input(prompt):
    while True:
        raw = input(prompt).strip()
        try:
            return float(raw)
        except ValueError:
            print("Please enter a numeric value.")


def main():
    print("ChurnGuard - Customer Churn Prediction")
    print("-" * 40)

    tenure = get_int_input("Enter tenure (months): ")
    monthly_charges = get_float_input("Enter Monthly Charges: ")
    total_charges = get_float_input("Enter Total Charges: ")
    senior_citizen = get_int_input("Senior Citizen? (1 = Yes, 0 = No): ", valid_values=[0, 1])
    contract = get_int_input(
        "Contract type (0 = Month-to-month, 1 = One year, 2 = Two year): ",
        valid_values=[0, 1, 2],
    )

    input_data = pd.DataFrame(
        [[tenure, monthly_charges, total_charges, senior_citizen, contract]],
        columns=FEATURES,
    )

    prediction = model.predict(input_data)[0]

    print("-" * 40)
    if prediction == 1:
        print("Prediction: This customer is likely to CHURN.")
    else:
        print("Prediction: This customer is likely to STAY.")


if __name__ == "__main__":
    main()
