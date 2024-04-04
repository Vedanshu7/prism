# Prism

End-to-end ML pipeline for tabular data — data ingestion, feature engineering, model training, and FastAPI serving.

## Features

- Data loading (CSV, JSON, Parquet) with schema validation
- Preprocessing: missing values, encoding, normalization, skew correction
- Feature engineering: interactions, polynomial features, selection
- Model training: RandomForest, GradientBoosting, LogisticRegression, SVM, ElasticNet
- Cross-validation and hyperparameter search
- Experiment tracking (MLflow-compatible)
- FastAPI model serving endpoint
- CLI: `prism train`, `prism evaluate`, `prism serve`

## Quick Start

```bash
pip install -r requirements.txt
prism train --config configs/example_classification.yaml
prism serve --model outputs/model.pkl --port 8000
```

## Pipeline

```
Data → Validate → Preprocess → Feature Engineering → Train → Evaluate → Serve
```
