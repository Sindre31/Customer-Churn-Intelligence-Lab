import pandas as pd


NUMERIC_COLUMNS = ["tenure_months", "monthly_charges", "total_charges", "support_tickets_last_3m"]
CATEGORICAL_COLUMNS = ["contract_type", "internet_service", "paperless_billing"]
TARGET_COLUMN = "churn"


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")

    for col in NUMERIC_COLUMNS:
        df[col] = df[col].fillna(df[col].median())

    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].fillna("Unknown")

    df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"Yes": 1, "No": 0})
    return df
