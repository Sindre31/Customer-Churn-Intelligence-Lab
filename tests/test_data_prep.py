import pandas as pd
from src.data_prep import clean_data


def test_clean_data_maps_target_and_fills_missing_values():
    df = pd.DataFrame(
        {
            "customer_id": [1, 2],
            "tenure_months": [12, None],
            "monthly_charges": [70.5, None],
            "total_charges": ["100.0", None],
            "support_tickets_last_3m": [1, None],
            "contract_type": ["Month-to-month", None],
            "internet_service": ["Fiber", None],
            "paperless_billing": ["Yes", None],
            "churn": ["Yes", "No"],
        }
    )

    cleaned = clean_data(df)

    assert cleaned["churn"].tolist() == [1, 0]
    assert cleaned.isna().sum().sum() == 0
