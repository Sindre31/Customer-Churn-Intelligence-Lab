# Customer Churn Intelligence Lab

Customer Churn Intelligence Lab is a portfolio data science project focused on predicting customer churn and presenting the results in a structured, business-facing way.

The repository is designed to show an end-to-end workflow rather than just a single notebook.

## Project purpose

This project demonstrates a typical supervised machine learning workflow:
- loading raw data
- cleaning and preprocessing
- feature engineering
- model training
- model evaluation
- exporting interpretable outputs

It is built as a portfolio project and uses a small demo dataset included in the repository.

## Problem

Customer churn is a common business problem in subscription and telecom-style companies. The goal of this project is to estimate which customers are more likely to churn and identify the factors most associated with that risk.

## What this project demonstrates

- organized Python project structure
- preprocessing with pandas and scikit-learn
- baseline model comparison
- evaluation using standard classification metrics
- feature importance export
- reproducible pipeline execution
- basic testing and CI

## Tech stack

- Python
- pandas
- NumPy
- scikit-learn
- matplotlib
- Pytest
- GitHub Actions

## Repository structure

- `src/` contains reusable pipeline code
- `data/raw/` contains the demo input dataset
- `data/processed/` stores cleaned data
- `outputs/` stores generated metrics and prediction samples
- `tests/` contains basic tests
- `run_pipeline.py` runs the full workflow

## Workflow

1. Load the raw telecom-style customer dataset
2. Clean missing values and map the target variable
3. Build preprocessing steps for numeric and categorical features
4. Train multiple classification models
5. Compare model performance
6. Save the best model
7. Export metrics, feature importance, and sample predictions

## Main outputs

- `outputs/metrics.json`
- `outputs/feature_importance.csv`
- `outputs/predictions_sample.csv`

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_pipeline.py
```

## Important note about the data and outputs

The dataset included in this repository is a small demo dataset created for portfolio purposes.

The exported files in `outputs/` should be understood as example outputs from the included sample data. They are useful for demonstrating workflow and repository structure, but they should not be interpreted as results from a large-scale production analysis.


## Possible next steps

- add cross-validation and hyperparameter tuning
- improve class imbalance handling
- add SHAP or permutation-based explanations
- create a dashboard for predictions and risk segments
- deploy the model behind an API
- replace the demo dataset with a larger public dataset
