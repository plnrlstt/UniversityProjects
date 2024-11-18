import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Step 1: Read and preprocess the data
df = pd.read_csv('MaleVectors.csv')

# Extract the words and vectors
words = df['word']
vectors = df.drop(columns=['word'])

# Standardize the vectors
scaler = StandardScaler()
scaled_vectors = scaler.fit_transform(vectors)

# Step 2: Apply PCA for Dimensionality Reduction
def apply_pca():
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(scaled_vectors)
    return pca_results

pca_results = apply_pca()

# Step 3: Perform K-Means Clustering
def perform_kmeans(n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(pca_results)
    return kmeans

# Elbow Method to find the optimal number of clusters
def elbow_plot():
    wcss = []
    max_clusters = 10
    for i in range(1, max_clusters + 1):
        kmeans = perform_kmeans(i)
        wcss.append(kmeans.inertia_)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_clusters + 1), wcss, marker='o')
    plt.title('Elbow Method for Optimal Number of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.show()

elbow_plot()

# Perform K-Means with the chosen number of clusters
n_clusters = 5  # Replace with the number of clusters determined from the elbow plot
kmeans = perform_kmeans(n_clusters)

# Step 4: Visualize the Clusters with PCA
def plot_pca_clusters():
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(pca_results[:, 0], pca_results[:, 1], c=kmeans.labels_, cmap='viridis', marker='o')

    # Add text labels to the scatter plot
    for i, word in enumerate(words):
        plt.text(pca_results[i, 0], pca_results[i, 1], word, fontsize=8, alpha=0.75)

    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title(f'PCA Visualization with K-Means Clustering ({n_clusters} Clusters)')
    plt.colorbar(scatter, label='Cluster Label')
    plt.show()

plot_pca_clusters()