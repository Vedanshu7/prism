---
title: Prism
type: Machine Learning
projectURL: prism
descriptionShort: An end-to-end ML pipeline for tabular data — ingestion, feature engineering, training, and FastAPI serving.
descriptionLong: Prism is an end-to-end machine learning pipeline for tabular data, taking a dataset from raw file to served model. It loads CSV, JSON, or Parquet with schema validation, then handles preprocessing — missing values, encoding, normalization, and skew correction — before generating interaction and polynomial features and running feature selection. Training covers RandomForest, GradientBoosting, LogisticRegression, SVM, and ElasticNet, with cross-validation, hyperparameter search, and MLflow-compatible experiment tracking. Trained models are served behind a FastAPI endpoint. The whole flow is driven from YAML config through a three-command CLI: train, evaluate, and serve.
viewCodeUrl: https://github.com/Vedanshu7/prism
viewProjectUrl:
projectImg:
technologies:
  - Python
  - scikit-learn
  - FastAPI
  - Pandas
  - NumPy
  - Docker
---
