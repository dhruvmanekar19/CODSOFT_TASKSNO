# Sales Prediction Using Python

## Project Description

This project uses machine learning to predict product sales based on advertising expenditure across different platforms.

The dataset contains advertising spending on:

- TV
- Radio
- Newspaper

The target variable is Sales.

## Objective

To build a machine learning model that predicts sales based on advertising expenditure.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Project Workflow

1. Load and explore the advertising dataset
2. Check for missing values
3. Analyze statistical information
4. Calculate correlations between variables
5. Visualize relationships between advertising and sales
6. Prepare features and target variables
7. Split the data into training and testing sets
8. Train a Linear Regression model
9. Evaluate the model using regression metrics
10. Predict sales for a new advertising budget
11. Visualize actual vs predicted sales

## Machine Learning Model

**Linear Regression**

The model uses TV, Radio, and Newspaper advertising expenditure as input features to predict Sales.

## Model Results

The model achieved:

- **MAE:** 1.275
- **MSE:** 2.908
- **RMSE:** 1.705
- **R² Score:** 0.906

The model achieved an R² score of approximately **90.6%** on the test dataset.

## Example Prediction

For an advertising budget of:

- TV: 150
- Radio: 25
- Newspaper: 20

The model predicted approximately:

**15.50 Sales**

## Visualizations

The project includes:

- TV Advertising vs Sales
- Radio Advertising vs Sales
- Newspaper Advertising vs Sales
- Actual vs Predicted Sales

## Author

Dhruv Manekar
