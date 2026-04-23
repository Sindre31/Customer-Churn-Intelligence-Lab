import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.features import build_preprocessor, split_xy


def train_models(df: pd.DataFrame):
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor()

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }

    results = {}
    best_name = None
    best_auc = -1
    best_pipeline = None

    for name, model in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model),
        ])
        pipeline.fit(X_train, y_train)
        probs = pipeline.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probs)

        results[name] = {
            "auc": float(round(auc, 4))
        }

        if auc > best_auc:
            best_auc = auc
            best_name = name
            best_pipeline = pipeline

    joblib.dump(best_pipeline, "models/best_model.pkl")
    return best_pipeline, best_name, results, X_test, y_test
