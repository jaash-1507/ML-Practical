import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Actual and predicted labels
actual = np.array(['Dog', 'Dog', 'Dog', 'Not Dog', 'Dog', 'Not Dog', 'Dog', 'Dog', 'Not Dog', 'Not Dog'])
predicted = np.array(['Dog', 'Not Dog', 'Dog', 'Not Dog', 'Dog', 'Dog', 'Dog', 'Dog', 'Not Dog', 'Not Dog'])

# Create the confusion matrix
cm = confusion_matrix(actual, predicted)

# Plot the confusion matrix as a heatmap
sns.heatmap(cm, annot=True, fmt='g', xticklabels=['Dog', 'Not Dog'], yticklabels=['Dog', 'Not Dog'])
plt.title('Confusion Matrix', fontsize=17, pad=20)
plt.xlabel('Predicted', fontsize=13)
plt.ylabel('Actual', fontsize=13)
plt.show()

# Print the classification report
print(classification_report(actual, predicted))
