# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv('sample_data.csv')

# Display the first few rows of the dataset
print(data.head())

# Check the columns to identify features and target variable
print(data.columns)

# Assuming 'Purchased' is the target variable, and 'Age', 'Salary' are features.
# Adjust these according to your actual dataset columns.
# If your dataset doesn't have a target variable, you need to create one or adjust the features accordingly.

# For demonstration, let's say 'Purchased' is the target variable:
# Encode the target variable if it's categorical
if 'Purchased' in data.columns:
    le = LabelEncoder()
    data['Purchased'] = le.fit_transform(data['Purchased'])

    # Define features (X) and target variable (y)
    X = data[['Age', 'Salary']]  # Replace with appropriate feature columns from your dataset
    y = data['Purchased']  # Target variable

    # Split the dataset into training (80%) and testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create the Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model using the training set
    model.fit(X_train, y_train)

    # Predict the outcomes for the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print(f"\nAccuracy: {accuracy}")
    print("\nConfusion Matrix:\n", conf_matrix)
    print("\nClassification Report:\n", class_report)

else:
    print("The dataset does not contain a 'Purchased' target variable. Please check your dataset.")
