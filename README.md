# K-means Clustering with Machine Learning

## Overview
This Python script demonstrates the application of K-means clustering, a machine learning algorithm used for unsupervised clustering tasks. K-means is employed here to partition a synthetic dataset into distinct clusters and visualize the results.
![image](https://github.com/mcoo2004/K-means-ML-Algorithm/assets/123424212/2647ef7c-be41-497a-aa19-4d8867ec856d)

## Libraries Used
- NumPy: for numerical computations
- Matplotlib: for plotting graphs
- scikit-learn: for implementing K-means clustering and generating synthetic datasets

## Usage
1. Ensure you have Python installed on your system.
2. Install the required libraries by running `pip install numpy matplotlib scikit-learn`.
3. Execute the script by running `python kmeans.py`.
4. The script generates two plots illustrating K-means clustering with different initialization methods: k-means++ and random.

## Functions
1. **create_dataset():** Generates a synthetic dataset using scikit-learn's make_blobs function.
2. **apply_kmeans(X, n_clusters, init_method):** Applies K-means clustering to the dataset with the specified number of clusters and initialization method.
3. **plot_clusters(X, kmeans, title):** Plots the clusters and centroids obtained from K-means clustering.

## Main Function
The main function orchestrates the execution of the script by creating the dataset, applying K-means with two different initialization methods, and plotting the results.

## Additional Notes
- K-means clustering is an unsupervised learning algorithm widely used for data clustering and segmentation tasks.
- The choice of initialization method (k-means++ or random) can impact the convergence and quality of the clustering results.
- This script serves as a basic demonstration of K-means clustering and can be extended for more complex datasets and analysis.

