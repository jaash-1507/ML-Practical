import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import kneighbors_graph
import networkx as nx

# Generate data and apply Spectral Clustering
X, _ = make_moons(n_samples=200, noise=0.1)
model = SpectralClustering(n_clusters=2, affinity='nearest_neighbors').fit(X)
labels = model.labels_

# Create nearest neighbors graph and convert to networkx
graph = kneighbors_graph(X, n_neighbors=10, mode='connectivity')
G = nx.from_scipy_sparse_array(graph)

# Plot nodes and edges
nx.draw(G, pos=X, node_color=labels, cmap='viridis', node_size=50, with_labels=False, edge_color='gray', alpha=0.5)
plt.title('Spectral Clustering with Nearest-Neighbor Connections')
plt.show()