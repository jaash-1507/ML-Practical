# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
# Generate synthetic data
X, _ = make_moons(n_samples=200, noise=0.1)
# Apply DBSCAN
model = DBSCAN(eps=0.2, min_samples=5).fit(X)
labels = model.labels_
# Plot the results
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.title('DBSCAN Clustering')
plt.show()
# Print clustering information
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
print(f"Number of Clusters Detected: {n_clusters}")
print(f"Number of Noise Points: {n_noise}")
