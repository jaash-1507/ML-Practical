# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Load the dataset
data = pd.read_csv('sample_data.csv')

# Create a binary target variable (1 if Salary > 80000, else 0)
data['SalaryAbove80K'] = (data['Salary'] > 80000).astype(int)

# Define the feature (Years of Experience) and the new binary target variable
X = data[['YearsExperience']]  # Independent variable
y = data['SalaryAbove80K']  # Binary target variable (0 or 1)

# Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Logistic Regression model
model = LogisticRegression()

# Train the model using the training set
model.fit(X_train, y_train)

# Predict the binary outcomes for the test set
y_pred = model.predict(X_test)

# Output the predicted values
print("\nPredicted Labels (0 = Salary ≤ 80000, 1 = Salary > 80000):", y_pred)
print("\nActual Labels (0 = Salary ≤ 80000, 1 = Salary > 80000):", y_test.values)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"\nAccuracy: {accuracy}")
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)

# Visualize the logistic regression results
plt.scatter(X_test, y_test, color='red', label="Actual")
plt.scatter(X_test, y_pred, color='blue', label="Predicted", marker='x')
plt.plot(X_test, model.predict_proba(X_test)[:, 1], color='green', label="Probability (Salary > 80000)")
plt.title('Logistic Regression (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Probability of Salary > 80000')
plt.legend()
plt.show()
