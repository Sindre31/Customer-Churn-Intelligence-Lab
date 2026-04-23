import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, roc_auc_score


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(round(accuracy_score(y_test, preds), 4)),
        "precision": float(round(precision_score(y_test, preds), 4)),
        "recall": float(round(recall_score(y_test, preds), 4)),
        "roc_auc": float(round(roc_auc_score(y_test, probs), 4)),
    }

    report = classification_report(y_test, preds, output_dict=True)
    return metrics, report, preds, probs


def extract_feature_importance(model, X_columns):
    preprocessor = model.named_steps["preprocessor"]
    trained_model = model.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()

    if hasattr(trained_model, "feature_importances_"):
        importances = trained_model.feature_importances_
    else:
        importances = abs(trained_model.coef_[0])

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    return df
