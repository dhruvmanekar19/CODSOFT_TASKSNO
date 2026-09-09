import pandas as pd

# Load the Titanic dataset
df = pd.read_csv("Titanic-Dataset.xls")

# Display the first 5 rows
print("First 5 rows of the dataset:")
print(df.head())

# Display dataset information
print("\nDataset Information:")
df.info()

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check survival distribution
print("\nSurvival Distribution:")
print(df["Survived"].value_counts())
# Data Preprocessing

# Fill missing values in Age with the median
df["Age"] = df["Age"].fillna(df["Age"].median())

# Fill missing values in Embarked with the most common value
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Drop columns that are not useful for prediction
df = df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)

# Convert categorical values into numerical values
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

print("\nDataset after preprocessing:")
print(df.head())

print("\nMissing values after preprocessing:")
print(df.isnull().sum())
# Separate features and target

X = df.drop("Survived", axis=1)
y = df["Survived"]

# Display features and target

print("\nFeatures:")
print(X.head())

print("\nTarget:")
print(y.head())
from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining data size:", X_train.shape)
print("Testing data size:", X_test.shape)
from sklearn.linear_model import LogisticRegression

# Create the model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

print("\nModel training completed successfully!")
from sklearn.metrics import accuracy_score, classification_report

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

# Display detailed performance report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Data Visualization

import matplotlib.pyplot as plt
import seaborn as sns

# Survival distribution
sns.countplot(x="Survived", data=df)

plt.title("Survival Distribution")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")

plt.show()
# Survival by Gender

sns.countplot(x="Sex", hue="Survived", data=df)

plt.title("Survival by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Passengers")

plt.show()
# Survival by Passenger Class

sns.countplot(x="Pclass", hue="Survived", data=df)

plt.title("Survival by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Number of Passengers")

plt.show()
# Confusion Matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()
# Predict survival for a new passenger

# Passenger details:
# Pclass, Sex, Age, SibSp, Parch, Fare, Embarked

new_passenger = pd.DataFrame([{
    "Pclass": 3,
    "Sex": 0,
    "Age": 25,
    "SibSp": 0,
    "Parch": 0,
    "Fare": 7.25,
    "Embarked": 0
}])

prediction = model.predict(new_passenger)

if prediction[0] == 1:
    print("\nPrediction: Passenger Survived")
else:
    print("\nPrediction: Passenger Did Not Survive")