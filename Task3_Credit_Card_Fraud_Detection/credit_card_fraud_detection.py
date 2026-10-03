import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score
)

from imblearn.over_sampling import SMOTE


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("creditcard.csv")

print("First 5 rows of the dataset:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())


# ============================================================
# 2. CLASS DISTRIBUTION
# ============================================================

print("\nClass Distribution:")
print(df["Class"].value_counts())

print("\nClass Distribution (%):")
print(df["Class"].value_counts(normalize=True) * 100)


# ============================================================
# 3. VISUALIZE CLASS IMBALANCE
# ============================================================

plt.figure(figsize=(7, 5))

sns.countplot(x="Class", data=df)

plt.title("Genuine vs Fraudulent Transactions")
plt.xlabel("Transaction Class")
plt.ylabel("Number of Transactions")
plt.xticks([0, 1], ["Genuine", "Fraudulent"])

plt.show()


# ============================================================
# 4. PREPARE FEATURES AND TARGET
# ============================================================

X = df.drop("Class", axis=1)
y = df["Class"]


# ============================================================
# 5. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data Size:", X_train.shape)
print("Testing Data Size:", X_test.shape)


# ============================================================
# 6. CREATE VALIDATION SET
# ============================================================

X_train, X_validation, y_train, y_validation = train_test_split(
    X_train,
    y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)

print("\nTraining Data After Validation Split:", X_train.shape)
print("Validation Data:", X_validation.shape)
print("Testing Data:", X_test.shape)


# ============================================================
# 7. NORMALIZE FEATURES
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_validation_scaled = scaler.transform(X_validation)
X_test_scaled = scaler.transform(X_test)

print("\nFeature normalization completed.")


# ============================================================
# 8. HANDLE CLASS IMBALANCE USING SMOTE
# ============================================================

print("\nTraining Class Distribution Before SMOTE:")
print(y_train.value_counts())

smote = SMOTE(random_state=42)

X_train_resampled, y_train_resampled = smote.fit_resample(
    X_train_scaled,
    y_train
)

print("\nTraining Class Distribution After SMOTE:")
print(pd.Series(y_train_resampled).value_counts())


# ============================================================
# 9. VISUALIZE CLASS DISTRIBUTION AFTER SMOTE
# ============================================================

plt.figure(figsize=(7, 5))

sns.countplot(x=y_train_resampled)

plt.title("Class Distribution After SMOTE")
plt.xlabel("Transaction Class")
plt.ylabel("Number of Transactions")
plt.xticks([0, 1], ["Genuine", "Fraudulent"])

plt.show()


# ============================================================
# 10. TRAIN LOGISTIC REGRESSION
# ============================================================

model = LogisticRegression(
    max_iter=1000
)

model.fit(X_train_resampled, y_train_resampled)

print("\nLogistic Regression model training completed successfully!")


# ============================================================
# 11. VALIDATION PROBABILITIES
# ============================================================

validation_probabilities = model.predict_proba(
    X_validation_scaled
)[:, 1]


# ============================================================
# 12. PRECISION-RECALL CURVE
# ============================================================

precision_values, recall_values, thresholds = precision_recall_curve(
    y_validation,
    validation_probabilities
)

average_precision = average_precision_score(
    y_validation,
    validation_probabilities
)

plt.figure(figsize=(8, 5))

plt.plot(
    recall_values,
    precision_values
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.grid()

plt.show()

print("\nAverage Precision Score:", average_precision)


# ============================================================
# 13. FIND BEST THRESHOLD USING VALIDATION DATA
# ============================================================

f1_scores = (
    2 * precision_values[:-1] * recall_values[:-1]
    / (precision_values[:-1] + recall_values[:-1] + 1e-10)
)

best_index = np.argmax(f1_scores)
best_threshold = thresholds[best_index]

print("\nBest Probability Threshold:", best_threshold)
print("Validation Precision:", precision_values[best_index])
print("Validation Recall:", recall_values[best_index])
print("Validation F1 Score:", f1_scores[best_index])


# ============================================================
# 14. APPLY THRESHOLD TO TEST DATA
# ============================================================

test_probabilities = model.predict_proba(
    X_test_scaled
)[:, 1]

y_pred = (
    test_probabilities >= best_threshold
).astype(int)


# ============================================================
# 15. FINAL MODEL EVALUATION
# ============================================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nFinal Model Evaluation:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


# ============================================================
# 16. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Genuine", "Fraudulent"]
    )
)


# ============================================================
# 17. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Genuine", "Fraudulent"],
    yticklabels=["Genuine", "Fraudulent"]
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.show()


# ============================================================
# 18. PRECISION, RECALL AND F1 VISUALIZATION
# ============================================================

metrics = {
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1
}

plt.figure(figsize=(8, 5))

plt.bar(
    metrics.keys(),
    metrics.values()
)

plt.ylim(0, 1)
plt.title("Fraud Detection Model Performance")
plt.ylabel("Score")

for i, value in enumerate(metrics.values()):
    plt.text(
        i,
        value + 0.02,
        f"{value:.3f}",
        ha="center"
    )

plt.show()


# ============================================================
# 19. FRAUD TRANSACTION AMOUNT DISTRIBUTION
# ============================================================

fraud_transactions = df[df["Class"] == 1]

plt.figure(figsize=(8, 5))

sns.histplot(
    fraud_transactions["Amount"],
    bins=50,
    kde=True
)

plt.title("Distribution of Fraudulent Transaction Amounts")
plt.xlabel("Transaction Amount")
plt.ylabel("Frequency")

plt.show()


# ============================================================
# 20. SAMPLE TRANSACTION PREDICTION
# ============================================================

sample_transaction = X_test.iloc[[0]]
actual_class = y_test.iloc[0]

sample_scaled = scaler.transform(
    sample_transaction
)

sample_probability = model.predict_proba(
    sample_scaled
)[0][1]

sample_prediction = int(
    sample_probability >= best_threshold
)

print("\nSample Transaction Prediction:")

if sample_prediction == 1:
    print("Prediction: FRAUDULENT TRANSACTION")
else:
    print("Prediction: GENUINE TRANSACTION")

print(
    "Actual Class:",
    "Fraudulent" if actual_class == 1 else "Genuine"
)

print(
    "Fraud Probability:",
    sample_probability
)

print(
    "Decision Threshold:",
    best_threshold
)


# ============================================================
# 21. PROJECT COMPLETION
# ============================================================

print(
    "\nCredit Card Fraud Detection project completed successfully!"
)