# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('sample_data.csv')

# Print column names to inspect them
print("Column names before cleanup:", data.columns)

# Clean up any possible leading/trailing spaces in column names
data.columns = data.columns.str.strip()

# Print column names after cleanup to confirm
print("Column names after cleanup:", data.columns)

# Define the feature (Years of Experience) and the target variable (Salary)
X = data[['YearsExperience']]  # Independent variable
y = data['Salary']  # Dependent variable

# Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Linear Regression model
model = LinearRegression()

# Train the model using the training set
model.fit(X_train, y_train)

# Predict the salaries for the test set
y_pred = model.predict(X_test)

# Output the test predictions
print("\nPredicted Salaries on Test Set:", y_pred)
print("\nActual Salaries in Test Set:", y_test.values)

# Calculate and print the Mean Squared Error and R-squared value
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nMean Squared Error: {mse}")
print(f"R-squared Value: {r2}")

# Visualize the training set results
plt.scatter(X_train, y_train, color='red', label="Actual (Train Data)")
plt.plot(X_train, model.predict(X_train), color='blue', label="Predicted (Regression Line)")
plt.title('Linear Regression (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.show()

# Visualize the test set results
plt.scatter(X_test, y_test, color='green', label="Actual (Test Data)")
plt.plot(X_train, model.predict(X_train), color='blue', label="Regression Line")
plt.title('Linear Regression (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.show()
