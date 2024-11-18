from steps.cli_helpers import get_valid_input, confirm_choice
import numpy as np
from collections import Counter

def count_clusters(cluster_labels):
    """
    Count the number of unique clusters, excluding noise points (labeled as -1).
    
    Args:
        cluster_labels (np.ndarray): Array of cluster labels assigned to each point.
    
    Returns:
        int: The number of clusters, excluding noise.
    """
    unique_clusters = np.unique(cluster_labels)
    return len(unique_clusters[unique_clusters != -1])

def find_largest_clusters(cluster_labels, max_clusters):
    """
    Identify the largest clusters by size, up to a specified maximum number.
    
    Args:
        cluster_labels (np.ndarray): Array of cluster labels assigned to each point.
        max_clusters (int): Maximum number of largest clusters to retrieve.
    
    Returns:
        list: Cluster IDs of the largest clusters, sorted by size.
    """
    cluster_counts = Counter(label for label in cluster_labels if label != -1)
    largest_clusters = [cluster_id for cluster_id, _ in cluster_counts.most_common(max_clusters)]
    return largest_clusters

def create_cluster_dict(cluster_labels, metadata, clusters_to_attack):
    """
    Create a dictionary mapping cluster IDs to metadata for each embedding in the cluster.
    
    Args:
        cluster_labels (np.ndarray): Array of cluster labels assigned to each point.
        metadata (list): Metadata associated with each embedding.
        clusters_to_attack (list): List of cluster IDs to include in the dictionary.
    
    Returns:
        dict: A dictionary where keys are cluster IDs and values are lists of metadata.
    """
    clusters = {}
    for cluster_id in clusters_to_attack:
        cluster_id = int(cluster_id)
        clusters[cluster_id] = []
        indices = np.where(cluster_labels == cluster_id)[0]
        for idx in indices:
            clusters[cluster_id].append(metadata[idx])
    return clusters

def display_cluster_details(clusters):
    """
    Print metadata for each cluster in the provided dictionary.
    
    Args:
        clusters (dict): Dictionary mapping cluster IDs to lists of metadata.
    
    Returns:
        None
    """
    for cluster_id, items in clusters.items():
        print(f"\nCluster ID: {cluster_id}")
        for item in items:
            print(item['caption'])

def select_clusters_to_attack(cluster_labels, metadata):
    """
    Guide the user in selecting clusters for analysis or attack. Includes input validation 
    and confirmation, then displays metadata for the selected clusters.
    
    Args:
        cluster_labels (np.ndarray): Array of cluster labels assigned to each point.
        metadata (list): Metadata associated with each embedding.
    
    Returns:
        dict or None: A dictionary of selected clusters and their metadata, or None if no clusters are found.
    """
    total_clusters = count_clusters(cluster_labels)

    if total_clusters == 0:
        return None

    while True:
        num_clusters_to_attack = int(get_valid_input(
            f"\nEnter the number of clusters to attack (1 - {total_clusters}): ",
            lambda value: value.isdigit() and 0 < int(value) <= total_clusters
        ))

        clusters_to_attack = find_largest_clusters(cluster_labels, num_clusters_to_attack)

        confirmed_clusters = confirm_choice(
            f"You selected {num_clusters_to_attack} clusters to attack. Is that correct?",
            clusters_to_attack
        )

        if confirmed_clusters:
            clusters = create_cluster_dict(cluster_labels, metadata, confirmed_clusters)
            display_cluster_details(clusters)
            return clusters
