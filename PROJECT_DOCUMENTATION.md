# Identify patients at risk of disease recurrence to support early healthcare intervention.
## 1. Project Overview

This project focuses on predicting the risk of disease recurrence using healthcare data. The goal is to build a machine learning model that identifies patients at high risk and supports early intervention and decision-making.

The workflow includes data cleaning, exploratory analysis, feature preparation, model training, and performance evaluation.

---

## 2. Dataset Description

The dataset contains patient healthcare features such as:

- Age
- BMI
- Medical indicators
- Risk labels (disease recurrence: Yes/No)

The data was cleaned and processed to remove missing values and prepare it for machine learning.

---

## 3. Data Preprocessing

Steps performed:

- Handling missing values
- Feature encoding
- Train-test split (80% training, 20% testing)
- Feature scaling and preparation

The dataset was divided into:

- Training set → used to teach the model
- Test set → used to evaluate performance

---

## 4. Model Training

A Logistic Regression model was trained using the training dataset.

The model learns patterns in patient data to predict recurrence risk.

Code used:

LogisticRegression(max_iter=1000)

---

## 5. Model Performance

### Accuracy

Model Accuracy: **97.37%**

This indicates the model predicts patient outcomes correctly in most cases.

---

### Confusion Matrix

[[70 1]
 [2 41]]

Interpretation:

- True Negatives: 70
- False Positives: 1
- False Negatives: 2
- True Positives: 41

The model rarely misses high-risk cases.

---

### Classification Report

Precision, recall, and F1-score show balanced performance:

- Precision: ~97–98%
- Recall: ~95–99%
- F1-score: ~96–98%

This indicates reliable and consistent predictions.

---

## 6. Key Insights

- The model shows strong predictive capability
- Very low false negative rate (important in healthcare)
- Suitable as a decision-support tool

---

## Key findings:

- Patients with higher BMI show increased recurrence risk

- Age is moderately correlated with recurrence

- Certain clinical indicators strongly predict outcomes