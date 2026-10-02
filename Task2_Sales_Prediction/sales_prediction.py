import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load the dataset
df = pd.read_csv("advertising.xls")


# Explore the dataset
print("First 5 rows of the dataset:")
print(df.head())

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nCorrelation Matrix:")
print(df.corr())


# Visualize relationships between advertising and sales

plt.figure(figsize=(8, 5))
sns.scatterplot(x="TV", y="Sales", data=df)
plt.title("TV Advertising vs Sales")
plt.xlabel("TV Advertising Spend")
plt.ylabel("Sales")
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(x="Radio", y="Sales", data=df)
plt.title("Radio Advertising vs Sales")
plt.xlabel("Radio Advertising Spend")
plt.ylabel("Sales")
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(x="Newspaper", y="Sales", data=df)
plt.title("Newspaper Advertising vs Sales")
plt.xlabel("Newspaper Advertising Spend")
plt.ylabel("Sales")
plt.show()


# Define features and target
X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]


# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining data size:", X_train.shape)
print("Testing data size:", X_test.shape)


# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel training completed successfully!")


# Make predictions
y_pred = model.predict(X_test)


# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R² Score:", r2)


# Predict sales for a new advertising budget
new_advertising = pd.DataFrame([{
    "TV": 150,
    "Radio": 25,
    "Newspaper": 20
}])

predicted_sales = model.predict(new_advertising)

print("\nNew Advertising Budget:")
print(new_advertising)

print("\nPredicted Sales:", predicted_sales[0])


# Visualize Actual vs Predicted Sales
plt.figure(figsize=(8, 5))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)

plt.show()