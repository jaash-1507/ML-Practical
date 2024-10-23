import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn import tree

data = pd.read_csv('decision_tree_data.csv')

print(data.head())

le = LabelEncoder()

data['Outlook'] = le.fit_transform(data['Outlook'])
data['Temperature'] = le.fit_transform(data['Temperature'])
data['Humidity'] = le.fit_transform(data['Humidity'])
data['Wind'] = le.fit_transform(data['Wind'])
data['Play'] = le.fit_transform(data['Play'])

X = data[['Outlook','Temperature','Humidity','Wind']]
y = data['Play']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

model = DecisionTreeClassifier()

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)
conf_matrix = confusion_matrix(y_test,y_pred)
class_report = classification_report(y_test,y_pred)

print(f"\nAccuracy: {accuracy}")
print("\nConfusion Matrix: \n",conf_matrix)
print("\nClassification Report:\n",class_report)

plt.figure(figsize=(12,8))
tree.plot_tree(model,feature_names=['Outlook','Temperature','Humidity','Wind'],class_names=['No','Yes'],filled=True)
plt.title("Decision Tree")
plt.show()