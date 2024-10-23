# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Sample dataset
data = {
    'Age': [22, 25, 47, 35, 46, 56, 23, 34, 45, 50, 23, 35, 42, 50],
    'Salary': [15000, 25000, 55000, 48000, 60000, 72000, 18000, 38000, 54000, 58000, 19000, 40000, 49000, 65000],
    'Purchased': ['No', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'Yes']
}

# Create a DataFrame
df = pd.DataFrame(data)

# Encode the target variable
df['Purchased'] = df['Purchased'].map({'No': 0, 'Yes': 1})

# Define features (X) and target (y)
X = df[['Age', 'Salary']]
y = df['Purchased']

# Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train the SVM model
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)

# Predict the outcomes for the test set
y_pred = svm_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print the evaluation results
print(f"Accuracy: {accuracy}")
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)

# Plot the decision boundary
plt.figure(figsize=(10, 6))

# Create a meshgrid for the feature space
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

# Predict the meshgrid points
Z = svm_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundaries
plt.contourf(xx, yy, Z, alpha=0.5, cmap='coolwarm')

# Plot the data points
sns.scatterplot(x=X_train[:, 0], y=X_train[:, 1], hue=y_train, palette='coolwarm', edgecolor='k')

# Label the plot
plt.title('SVM Decision Boundary', fontsize=16)
plt.xlabel('Age (scaled)', fontsize=12)
plt.ylabel('Salary (scaled)', fontsize=12)
plt.show()
