from src.data_prep import load_data, clean_data
from src.train import train_models
from src.evaluate import evaluate_model, extract_feature_importance
from src.utils import ensure_dirs, save_json


def main():
    ensure_dirs()

    df = load_data("data/raw/telecom_churn_sample.csv")
    df = clean_data(df)
    df.to_csv("data/processed/cleaned_churn_data.csv", index=False)

    model, best_name, train_results, X_test, y_test = train_models(df)
    metrics, report, preds, probs = evaluate_model(model, X_test, y_test)

    save_json(
        {
            "best_model": best_name,
            "model_comparison": train_results,
            "test_metrics": metrics,
        },
        "outputs/metrics.json",
    )

    feature_importance = extract_feature_importance(model, X_test.columns)
    feature_importance.to_csv("outputs/feature_importance.csv", index=False)

    predictions = X_test.copy()
    predictions["actual_churn"] = y_test.values
    predictions["predicted_churn"] = preds
    predictions["churn_probability"] = probs
    predictions.head(50).to_csv("outputs/predictions_sample.csv", index=False)

    print("Pipeline completed successfully.")
    print(f"Best model: {best_name}")
    print(f"Metrics: {metrics}")


if __name__ == "__main__":
    main()
