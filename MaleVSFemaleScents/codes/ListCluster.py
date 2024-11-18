import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Step 1: Read and preprocess the data
df = pd.read_csv('FemaleVectors.csv')

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

# Perform K-Means with a chosen number of clusters
n_clusters = 20  # Replace with the number of clusters you want
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
    plt.title(f'Female Scents: PCA Visualization with K-Means Clustering ({n_clusters} Clusters)')
    plt.colorbar(scatter, label='Cluster Label')
    plt.show()

plot_pca_clusters()

# Step 5: Save clusters as a list with separation
def save_clusters():
    cluster_dict = {}
    for word, label in zip(words, kmeans.labels_):
        if label not in cluster_dict:
            cluster_dict[label] = []
        cluster_dict[label].append(word)

    with open('NewFemaleClusters.txt', 'w') as file:
        for label, cluster_words in cluster_dict.items():
            file.write(f"Cluster {label}:\n")
            file.write('\n'.join(cluster_words))
            file.write('\n\n')  # Add a blank line between clusters

save_clusters()
