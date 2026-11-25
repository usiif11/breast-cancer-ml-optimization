
# Breast Cancer Classification & ML Optimization

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-2563EB?style=flat-square)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)
![Research](https://img.shields.io/badge/Project-Research-64748B?style=flat-square)

## Overview

A machine learning research project that combines Genetic Algorithms (GA) and Random Forest classification to investigate the impact of feature selection and hyperparameter optimization on breast cancer classification performance.

The main objective is to improve classification accuracy while reducing the number of input features.

## Key Results

| Metric | Baseline | Optimized Model |
|---|---:|---:|
| Test Accuracy | 95.61% | 98.25% |
| ROC-AUC | 0.9937 | 0.9937 |
| Number of Features | 30 | 9 |
| Feature Reduction | — | 70% |

The reported optimized model achieved:

- 2.64 percentage points improvement in test accuracy.
- 70% reduction in input features.
- ROC-AUC of 0.9937.

These results are reported from the original experiment and have not been independently reproduced.

## Dataset

Breast Cancer Wisconsin (Diagnostic) Dataset, available through Scikit-learn.

- Total samples: 569
- Original numerical features: 30
- Classification task: Benign vs. Malignant
- Problem type: Binary classification

## Project Methodology

1. Dataset loading and preparation.
2. Genetic Algorithm-based feature selection.
3. Random Forest classification.
4. Hyperparameter optimization.
5. Stratified 5-Fold Cross Validation.
6. Performance evaluation and comparison.

## Genetic Algorithm Configuration

| Parameter | Value |
|---|---:|
| Population Size | 50 |
| Generations | 20 |
| Mutation Rate | 0.05 |
| Independent Mutation Probability | 0.02 |
| Cross-Validation Folds | 5 |
| Fitness Metric | Mean Classification Accuracy |

Each feature selection chromosome represents a binary vector, where 1 indicates an included feature and 0 indicates an excluded feature.

## Selected Features

The optimization process selected 9 out of 30 original features:

- Mean Texture
- Mean Concave Points
- Mean Symmetry
- Mean Fractal Dimension
- Area Error
- Worst Texture
- Worst Area
- Worst Smoothness
- Worst Concavity

## Optimized Hyperparameters

| Hyperparameter | Value |
|---|---:|
| n_estimators | 474 |
| max_depth | 24 |
| min_samples_split | 3 |
| min_samples_leaf | 4 |
| max_features | 0.155 |

## Technologies

- Python
- Scikit-learn
- Random Forest
- Genetic Algorithms
- Jupyter Notebook
- Feature Selection
- Hyperparameter Optimization
- Stratified Cross Validation

## Repository Contents

- Jupyter Notebook: Original implementation and experiments.
- PDF Report: Supporting project documentation.
- README.md: Project overview and reported results.

## Future Improvements

- Reproduce the experiments in a clean environment.
- Verify feature selection and hyperparameter tuning against data leakage.
- Compare performance across additional classifiers.
- Evaluate precision, recall, F1-score, and confusion matrices.
- Investigate model interpretability using SHAP.
- Evaluate generalization on independent datasets.

## Disclaimer

This project was developed for educational and research purposes. It is not a clinically validated diagnostic system and should not be used for medical decision-making.

## Author

**Yousef Hossam**

AI Engineer | Machine Learning | Data Analysis

[GitHub](https://github.com/usiif11) | [LinkedIn](https://www.linkedin.com/in/yousef-hossam-63b167358)
