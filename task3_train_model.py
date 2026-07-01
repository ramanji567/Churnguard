"""
ChurnGuard - Task 3: Train a Classification Model
--------------------------------------------------
Loads and cleans the dataset (Task 2 steps), encodes it, splits it into
train/test sets, trains a Logistic Regression model, and evaluates how
well it predicts customer churn.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


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

# 2. Encode the target column: Yes -> 1, No -> 0
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# 3. Encode categorical columns with one-hot encoding
categorical_cols = [
    "gender",
    "PhoneService",
    "InternetService",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Drop any rows that still contain missing values (e.g. InternetService NaNs)
# so the model can train cleanly.
df = df.dropna()

# 4. Separate the data into X and y
X = df.drop(columns=["Churn"])
y = df["Churn"]

# 5. Split into train and test sets - 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Train a Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 7. Print the accuracy score on the test set
y_pred = model.predict(X_test)
print("=" * 60)
print("ACCURACY SCORE")
print("=" * 60)
print(accuracy_score(y_test, y_pred))

# 8. Print the classification report
print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)
print(classification_report(y_test, y_pred, target_names=["Stay", "Churn"]))
