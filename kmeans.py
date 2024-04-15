# Import libraries

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Create dataset
def create_dataset():
    X, y = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)
    return X

# Apply K-means
def apply_kmeans(X, n_clusters=3, init_method='k-means++'):
    kmeans = KMeans(n_clusters=n_clusters, init=init_method, random_state=0)
    kmeans.fit(X)
    return kmeans

# Plot clusters
def plot_clusters(X, kmeans, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis', marker='o', s=50, alpha=0.8)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='x', label='Centers')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# Main function
def main():
    X = create_dataset()
    kmeans_plus = apply_kmeans(X, init_method='k-means++')
    kmeans_random = apply_kmeans(X, init_method='random')
    plot_clusters(X, kmeans_plus, "K-means Clustering with k-means++ Initialization")
    plot_clusters(X, kmeans_random, "K-means Clustering with Random Initialization")

if __name__ == '__main__':
    main()
    