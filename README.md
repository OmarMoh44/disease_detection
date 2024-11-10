# disease_detection


## Project Overview

This project demonstrates a complete pipeline for a machine learning classification task, covering data preprocessing, model selection, and training. Three models are explored and compared: K-Nearest Neighbors (KNN), Logistic Regression, and a Deep Learning model built with Keras. Each model is evaluated on multiple metrics, including accuracy, precision, recall, and F1 score.

## Table of Contents

- [disease\_detection](#disease_detection)
  - [Project Overview](#project-overview)
  - [Table of Contents](#table-of-contents)
  - [1. Preprocessing](#1-preprocessing)
  - [2. Model Selection](#2-model-selection)
  - [3. Evaluation Criteria](#3-evaluation-criteria)
  - [4. Comparison Between Approaches](#4-comparison-between-approaches)
  - [Explanation of Model Choices](#explanation-of-model-choices)
  - [Conclusion](#conclusion)

## 1. Preprocessing

The data preparation process includes several steps to ensure consistency and accuracy in model training:

- **Data Handling**: Data is loaded from `student_version.csv`. Features (X) and target variable (y) are separated.
- **Feature Types**: Features are divided into numerical and categorical types.
- **Feature Selection**: Numerical features with fewer than 8 unique values are reclassified as categorical.
- **Outlier Handling**: Outliers are identified using the interquartile range (IQR) method and replaced with mean values.
- **Scaling and Encoding**: `StandardScaler` is applied to numerical features, and `OneHotEncoder` is applied to categorical features. A `ColumnTransformer` ensures consistent transformations across all features.

## 2. Model Selection

Three models are tested to assess their performance and suitability for the dataset:

- **K-Nearest Neighbors (KNN)**: A non-parametric, distance-based classifier.
  - *Hyperparameters*: 5 neighbors, Minkowski distance with p=2 (Euclidean).
- **Logistic Regression**: A classic model for binary classification, suitable for linearly separable data.
- **Deep Learning (Keras)**: A neural network with three layers.
  - *Architecture*: Two hidden layers with 16 and 8 neurons, ReLU activation, and a sigmoid output layer for binary classification.

## 3. Evaluation Criteria

Models are evaluated using multiple metrics to gauge performance:

- **Accuracy**: Proportion of correctly classified instances.
- **Precision**: Ratio of true positives to all positive predictions.
- **Recall**: Ratio of true positives to actual positives.
- **F1 Score**: Harmonic mean of precision and recall.
- **Additional Metrics for Deep Learning**: Training loss and accuracy are tracked over epochs.

## 4. Comparison Between Approaches

| Model               | Pros                                                   | Cons                                                     | Accuracy |
|---------------------|--------------------------------------------------------|----------------------------------------------------------|----------|
| **KNN**             | Simple and effective for small datasets                | Sensitive to irrelevant features and data scaling        | 88.6%    |
| **Logistic Regression** | Interpretable and efficient for binary tasks          | Assumes linearity, limited on complex data               | 89.6%    |
| **Deep Learning (Keras)** | Captures complex relationships, flexible              | Requires more data, prone to overfitting                 | 88.04%   |

## Explanation of Model Choices

Each model was selected for specific advantages:

- **KNN**: Effective for datasets without assumptions on underlying data distributions.
- **Logistic Regression**: Useful for understanding feature importance and effective for binary classification.
- **Deep Learning**: Capable of capturing non-linear patterns in complex datasets.

## Conclusion

This project demonstrates the end-to-end development of a machine learning model, including preprocessing, model selection, and evaluation. Each model offers distinct benefits, with Logistic Regression achieving the highest accuracy.
