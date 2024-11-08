import numpy as np
import os
from sklearn.cluster import DBSCAN
import umap.umap_ as umap
import matplotlib.pyplot as plt
from collections import Counter

# Load all embeddings into lists
def load_embeddings(embedding_dir):
    print("Reading embeddings...")
    embeddings = []
    urls = []
    captions = []
    for filename in os.listdir(embedding_dir):
        if filename.endswith('.npz'):
            data = np.load(os.path.join(embedding_dir, filename))
            embeddings.append(data['embedding'])  # 512-dimensional vector
            if 'url' in data: urls.append(data['url'])
            captions.append(data['caption'])
    return np.array(embeddings).squeeze(), urls, captions

# Validate embeddings

def validate_embeddings_shape(embeddings, expected_dim=512):
    if embeddings.shape[1] != expected_dim:
        raise ValueError(f"Embeddings should have shape (n_samples, {expected_dim}). Current shape is {embeddings.shape}")

# Cluster embeddings using DBSCAN
def cluster_embeddings(embeddings, eps=0.5, min_samples=2):
    print("Finding clusters...")
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    cluster_labels = clustering.fit_predict(embeddings)
    return cluster_labels

# Display URLs and captions for images in selected clusters
def display_selected_clusters_info(cluster_labels, urls, captions, cluster_count):
    unique_clusters = np.unique(cluster_labels)
    unique_clusters = unique_clusters[unique_clusters != -1]  # Ignore noise points (label -1)

    # Count the number of points in each cluster
    cluster_sizes = Counter(cluster_labels[cluster_labels != -1])
    largest_clusters = [cluster for cluster, _ in cluster_sizes.most_common(cluster_count)]

    for cluster_id in largest_clusters:
        print(f"\nCluster ID: {cluster_id}")
        indices_in_cluster = np.where(cluster_labels == cluster_id)[0]

        for idx in indices_in_cluster:
            if urls != []:
                print(f"  URL: {urls[idx]}")
            print(f"  Caption: {captions[idx]}")

# Main run function to execute all steps
def run(embedding_dir):
    embeddings, urls, captions = load_embeddings(embedding_dir)
    validate_embeddings_shape(embeddings)
    cluster_labels = cluster_embeddings(embeddings)
    
    # Find and display number of clusters
    unique_clusters = np.unique(cluster_labels)
    num_clusters = len(unique_clusters[unique_clusters != -1])  # Ignore noise points (label -1)
    print(f"Total clusters found (excluding noise): {num_clusters}")

    # Ask user how many clusters they want to see
    cluster_count = int(input(f"Enter the number of clusters to display (1-{num_clusters}): "))
    cluster_count = max(1, min(cluster_count, num_clusters))  # Ensure valid input

    display_selected_clusters_info(cluster_labels, urls, captions, cluster_count)

if __name__ == "__main__":
    embedding_dir = "../data/clip_embeddings/hahminlew_kream-product-blip-captions"
    run(embedding_dir)
