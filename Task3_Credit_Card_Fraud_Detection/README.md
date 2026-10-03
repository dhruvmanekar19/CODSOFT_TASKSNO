# Credit Card Fraud Detection

## Project Overview

This project focuses on detecting fraudulent credit card transactions using machine learning.

The dataset contains highly imbalanced transaction data, where fraudulent transactions represent only a small portion of the total transactions. To address this class imbalance, SMOTE (Synthetic Minority Over-sampling Technique) was applied to the training data.

A Logistic Regression model was then trained and evaluated using precision, recall, F1-score, and a confusion matrix.

---

## Objective

Build a machine learning model capable of identifying fraudulent credit card transactions while handling the severe class imbalance present in the dataset.

---

## Dataset

The project uses the Credit Card Fraud Detection dataset containing:

- 284,807 transactions
- 30 input features
- 1 target variable (`Class`)
- `Class = 0` → Genuine transaction
- `Class = 1` → Fraudulent transaction

The dataset includes transaction time, transaction amount, and PCA-transformed features (`V1` to `V28`).

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn

---

## Machine Learning Workflow

1. Load and inspect the dataset
2. Analyze class distribution
3. Check missing values and duplicate records
4. Split the data into training, validation, and testing sets
5. Normalize the features using StandardScaler
6. Handle class imbalance using SMOTE
7. Train a Logistic Regression classification model
8. Generate probability predictions
9. Analyze the Precision-Recall curve
10. Select a probability threshold using validation data
11. Evaluate the final model on the untouched test set
12. Analyze the confusion matrix
13. Predict a sample transaction

---

## Handling Class Imbalance

The dataset contains significantly more genuine transactions than fraudulent transactions.

SMOTE was applied only to the training data to create additional synthetic examples of the minority class.

### Training distribution before SMOTE

- Genuine: 181,961
- Fraudulent: 315

### Training distribution after SMOTE

- Genuine: 181,961
- Fraudulent: 181,961

The validation and test datasets were kept separate from the oversampling process.

---

## Model

### Logistic Regression

Logistic Regression was used as the classification algorithm for distinguishing between genuine and fraudulent transactions.

The model was trained on the SMOTE-balanced training data.

---

## Threshold Optimization

Instead of relying only on the default probability threshold, a Precision-Recall analysis was performed on the validation set.

The threshold that produced the highest validation F1-score was selected and then applied to the untouched test dataset.

### Selected threshold

`0.9999995179`

### Validation Results

- Precision: 0.829
- Recall: 0.734
- F1-score: 0.779

---

## Final Model Results

The final model was evaluated on the test dataset.

| Metric | Score |
|---|---:|
| Accuracy | 99.94% |
| Precision | 81.00% |
| Recall | 82.65% |
| F1-score | 81.82% |
| Average Precision | 0.6814 |

For fraud detection, precision, recall, and F1-score are emphasized because the dataset is highly imbalanced.

---

## Confusion Matrix

```text
[[56845    19]
 [   17    81]]
