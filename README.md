# Parkinson's Disease Detection using Machine Learning

This project aims to build and evaluate machine learning models for detecting Parkinson's Disease (PD) from speech features. The dataset used for this analysis contains vocal features extracted from individuals diagnosed with Parkinson's Disease and healthy individuals. The goal is to develop predictive models using various supervised learning techniques.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Models and Algorithms](#models-and-algorithms)
- [Evaluation](#evaluation)
- [Installation](#installation)
- [Usage](#usage)
- [References](#references)
- [Acknowledgments](#acknowledgments)

## Project Overview

Parkinson's Disease is a progressive neurological disorder that affects movement and speech. Early detection is crucial for improving treatment outcomes. In this project, we apply machine learning techniques to detect Parkinson's Disease from vocal features such as jitter, shimmer, and harmonic-to-noise ratio (HNR), among others.

The dataset used in this analysis is provided by the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/parkinsons).

## Dataset

The dataset contains 195 samples with 23 features related to vocal measurements. The features include:

- **MDVP:Fo (mean fundamental frequency)**
- **MDVP:Jitter**
- **MDVP:Shimmer**
- **NHR (Noise-to-Harmonics Ratio)**
- **HNR (Harmonics-to-Noise Ratio)**
- **RPDE, DFA, spread1, spread2, etc.**

The target variable (`status`) indicates whether the individual has Parkinson's Disease (1) or is healthy (0).

For more details on the dataset, please refer to the [UCI Parkinson's dataset page](https://archive.ics.uci.edu/ml/datasets/parkinsons).

## Methodology

The project follows the following steps:

1. **Data Acquisition and Ingestion**: The dataset is loaded into a pandas DataFrame for easy manipulation.
2. **Data Preprocessing**: This includes renaming columns for clarity, handling missing data, and identifying and handling outliers.
3. **Feature Selection**: Features were selected based on their relevance to Parkinson's Disease prediction.
4. **Data Splitting**: The data is split into training (80%) and testing (20%) sets.
5. **Model Training**: Supervised learning models, including classification and regression algorithms, are trained on the data.
6. **Model Evaluation**: The models are evaluated using various metrics such as accuracy, precision, recall, F1-score (for classification) and RMSE, MAE, R-squared (for regression).

## Models and Algorithms

The following models were evaluated:

### Classification Algorithms:
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- Gradient Boosting
- XGBoost
- AdaBoost

### Regression Algorithms:
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Support Vector Regressor
- Gradient Boosting Regressor
- XGBoost Regressor
- AdaBoost Regressor

## Evaluation

The performance of each model was evaluated using the following metrics:

### Classification Metrics:
- Accuracy
- AUC score
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC Curve

### Regression Metrics:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R-squared (R2)

The top-performing models for classification were **XGBoost**, **Random Forest**, and **Gradient Boosting**. For regression tasks, **AdaBoost Regressor** had the lowest test RMSE and MAE.
