# 📉 ChurnGuard — Customer Churn Prediction Pipeline

A complete, end-to-end data science mini-project: exploring a **messy, real-world telecom dataset**, cleaning it with a reproducible pipeline, training a **Logistic Regression** churn classifier, and shipping a simple interactive tool that gives an instant churn prediction for a single customer.

Built as a 4-stage pipeline that mirrors how churn-prediction work actually happens in industry: **explore → clean → model → deploy (as a lightweight tool)**.

---

## 🧭 Project Overview

| Stage | Script | What it does |
|---|---|---|
| 1️⃣ Explore | [`task1_load_explore.py`](task1_load_explore.py) | Loads the raw data and profiles it — shape, dtypes, missing values, duplicates, inconsistent categories |
| 2️⃣ Clean | [`task2_clean_data.py`](task2_clean_data.py) | Fixes whitespace, casing, typos, junk values, outliers, and missing data with a documented, repeatable pipeline |
| 3️⃣ Model | [`task3_train_model.py`](task3_train_model.py) | Encodes features, splits train/test, trains Logistic Regression, reports accuracy + classification report |
| 4️⃣ Predict | [`task4_predict.py`](task4_predict.py) | Retrains on the full cleaned dataset and takes live user input to predict churn risk for a single customer |

---

## 🗂 Dataset

`churnguard_data.csv` — a synthetic telecom customer dataset (1,030 rows) collected from multiple regional offices, intentionally messy to simulate real-world data quality problems.

| Column | Description |
|---|---|
| `customerID` | Unique customer ID *(dropped — not predictive)* |
| `gender` | Customer gender |
| `SeniorCitizen` | 1 if senior citizen, 0 otherwise |
| `tenure` | Months with the company |
| `PhoneService` | Has phone service (Yes/No) |
| `InternetService` | Type of internet service |
| `Contract` | Contract type |
| `PaperlessBilling` | Paperless billing (Yes/No) |
| `PaymentMethod` | Payment method |
| `MonthlyCharges` | Monthly bill amount |
| `TotalCharges` | Total amount billed to date |
| `Churn` | Target — did the customer cancel? |

### Known data quality issues (found in Task 1, fixed in Task 2)
- Extra leading/trailing whitespace (`gender`, `PaymentMethod`)
- Inconsistent casing (`YES`, `yEs`, `nO`, etc.) in `PhoneService`, `PaperlessBilling`, `Churn`
- Typos & alternate spellings in `Contract` (`Monthly`, `1 year`, `month to month`...) and `InternetService` (`Fibre optic`, `FiberOptic`, `dsl`...)
- `TotalCharges` stored as text with junk values (`N/A`, `--`, `?`, blanks)
- Missing values across `tenure`, `InternetService`, `MonthlyCharges`, `TotalCharges`
- Impossible / outlier values — negative `tenure`, `MonthlyCharges` under \$10 or over \$200
- 30 fully duplicated rows

---

## ⚙️ Cleaning Pipeline (Task 2)

1. Drop `customerID`
2. Remove duplicate rows
3. Strip whitespace from `gender`, `PaymentMethod`
4. Standardise casing on `Churn`, `PhoneService`, `PaperlessBilling` → Title Case
5. Normalise `Contract` → `Month-to-month` / `One year` / `Two year`
6. Normalise `InternetService` → `DSL` / `Fiber optic` / `No`
7. Convert `TotalCharges` to numeric, coercing junk to `NaN`
8. Drop rows with `tenure <= 0`
9. Drop rows with `MonthlyCharges < 10` or `> 200`
10. Impute missing values — `MonthlyCharges`/`TotalCharges` → column mean, `tenure` → column median (rounded to int)

Result: a clean **980-row** DataFrame ready for modelling.

---

## 🤖 Model (Task 3)

- **Algorithm:** Logistic Regression (`max_iter=1000`)
- **Encoding:** `Churn` → binary (Yes=1/No=0); categorical features one-hot encoded with `drop_first=True`
- **Split:** 80% train / 20% test, `random_state=42`

**Results on the held-out test set:**

| Metric | Score |
|---|---|
| Accuracy | ~0.69 |
| Precision (Churn) | ~0.55 |
| Recall (Churn) | ~0.27 |
| F1 (Churn) | ~0.36 |

> A solid baseline. Class imbalance (more "Stay" than "Churn") limits recall on the churn class — see [Next Steps](#-next-steps) for how this would be improved in production.

---

## 🔮 Interactive Prediction Tool (Task 4)

`task4_predict.py` retrains the model on the **entire cleaned dataset** using five numeric/ordinal features (`tenure`, `MonthlyCharges`, `TotalCharges`, `SeniorCitizen`, `Contract`) so a non-technical user (e.g. a customer success rep) can type in a customer's details and get an instant answer:

```
$ python task4_predict.py
ChurnGuard - Customer Churn Prediction
----------------------------------------
Enter tenure (months): 5
Enter Monthly Charges: 95.50
Enter Total Charges: 450.00
Senior Citizen? (1 = Yes, 0 = No): 0
Contract type (0 = Month-to-month, 1 = One year, 2 = Two year): 0
----------------------------------------
Prediction: This customer is likely to CHURN.
```

---

## 🚀 Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/churnguard.git
cd churnguard

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the pipeline in order
python task1_load_explore.py   # explore the raw data
python task2_clean_data.py     # clean it
python task3_train_model.py    # train & evaluate
python task4_predict.py        # interactive prediction
```

**Requirements:** Python 3.9+, `pandas`, `scikit-learn`

---

## 🛠 Tech Stack

`Python` · `pandas` · `scikit-learn` · Logistic Regression

---

## 📌 Next Steps

Ideas for extending this project further:
- Handle class imbalance (`class_weight="balanced"`, SMOTE) to improve churn recall
- Try tree-based models (Random Forest, XGBoost) and compare via ROC-AUC
- Add cross-validation and hyperparameter tuning
- Wrap `task4_predict.py` in a small Streamlit/Flask app for a real UI
- Add unit tests for the cleaning pipeline (`pytest`)

---

## 📁 Repo Structure

```
churnguard/
├── churnguard_data.csv
├── task1_load_explore.py
├── task2_clean_data.py
├── task3_train_model.py
├── task4_predict.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 👤 Author

*Your Name* — feel free to connect on [LinkedIn](#) or check out more of my work on [GitHub](#).

