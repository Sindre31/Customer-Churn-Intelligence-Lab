# Customer Churn Intelligence Lab

I built this project to practice the full workflow of a churn prediction problem, from raw data and preprocessing to model evaluation and exported outputs. I wanted the repository to feel closer to a small reusable project than a single notebook submission.

## Why I built this

I wanted to practice building a cleaner machine learning repository outside of a notebook-only workflow.

My focus was:
- preprocessing mixed feature types
- comparing baseline classification models
- exporting results into files that would be easier to use in a business setting

## Project purpose

I organized this project as a small end-to-end churn workflow instead of keeping everything in one notebook.

It is built as a portfolio project and uses a small demo dataset included in the repository.

## Problem

Customer churn is a common business problem in subscription and telecom-style companies. The goal of this project is to estimate which customers are more likely to churn and identify the factors most associated with that risk.

## What I implemented

In this project, I implemented:
- preprocessing for numeric and categorical features
- a train/evaluate pipeline for churn classification
- baseline model comparison
- output export to metrics and CSV files
- a project structure with reusable code under `src/`
- basic tests and pipeline execution from `run_pipeline.py`

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

## Challenges and tradeoffs

A main tradeoff in this project was keeping the pipeline simple and readable while still showing a realistic ML workflow.

I chose a small demo dataset and straightforward baseline models so the repository would stay easy to understand and run locally, even if that meant limiting the depth of the analysis.

## Why this is in my portfolio

I included this project because it shows how I approach structuring a machine learning project beyond a single notebook. It demonstrates preprocessing, model comparison, reproducible execution, exported outputs, and code organized into reusable parts.

## Tech stack

- Python
- pandas
- NumPy
- scikit-learn
- matplotlib
- Pytest
- GitHub Actions

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_pipeline.py
```

## Notes on the data and outputs

The dataset included in this repository is a small demo dataset created for portfolio purposes.

The exported files in `outputs/` should be understood as example outputs from the included sample data. They are useful for demonstrating workflow and repository structure, but they should not be interpreted as results from a large-scale production analysis.

## Limitations

- the dataset is small and intended for demonstration
- model evaluation is limited compared with a production workflow
- explanations are simplified and meant for portfolio presentation

## Personal note

I used this project to practice turning a standard ML exercise into a cleaner repository with reusable code, outputs, and clearer documentation.

## Possible next steps

- add cross-validation and hyperparameter tuning
- improve class imbalance handling
- add SHAP or permutation-based explanations
- create a dashboard for predictions and risk segments
- deploy the model behind an API
- replace the demo dataset with a larger public dataset
