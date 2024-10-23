# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
# Load dataset
data = load_iris()
X = data.data
# Apply PCA to reduce dimensions to 2
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
# Plot the results
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=data.target, cmap='viridis')
plt.title('PCA Reduction of Iris Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
# Print PCA information
explained_variance = pca.explained_variance_ratio_
print(f"Explained Variance by Components: {explained_variance}")
print(f"Shape of Reduced Data: {X_reduced.shape}")
