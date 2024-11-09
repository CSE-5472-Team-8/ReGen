from CLI.helpers import get_valid_input, confirm_choice
import numpy as np
from collections import Counter

def count_clusters(cluster_labels):
    """Returns the number of clusters excluding noise points."""
    unique_clusters = np.unique(cluster_labels)
    return len(unique_clusters[unique_clusters != -1])  # Ignore noise points (label -1)

def find_largest_clusters(cluster_labels, max_clusters):
    """Finds the largest clusters by size, up to the specified number."""
    cluster_counts = Counter(label for label in cluster_labels if label != -1)
    largest_clusters = [cluster_id for cluster_id, _ in cluster_counts.most_common(max_clusters)]
    return largest_clusters

def create_cluster_dict(cluster_labels, metadata, clusters_to_display):
    """Creates a dictionary with metadata for each embedding in the specified clusters."""
    clusters = {}
    for cluster_id in clusters_to_display:
        cluster_id = int(cluster_id)
        clusters[cluster_id] = []
        indices = np.where(cluster_labels == cluster_id)[0]
        for idx in indices:
            clusters[cluster_id].append(metadata[idx])
    return clusters

def display_cluster_details(clusters):
    """Displays metadata for each cluster in the provided dictionary."""
    for cluster_id, items in clusters.items():
        print(f"\nCluster ID: {cluster_id}")
        for item in items:
            print(item)

def select_clusters_to_attack(cluster_labels, metadata):
    """Guides the user through selecting clusters to display, with input validation and confirmation."""
    
    # Get total clusters (excluding noise) and prompt for how many to display
    total_clusters = count_clusters(cluster_labels)

    if total_clusters == 0:
        print("No clusters found. This model is likely invulnerable to model inversion attacks.")
        return None

    while True:
        num_clusters_to_display = int(get_valid_input(
            f"\nEnter the number of clusters to display (0 - {total_clusters}): ",
            lambda value: value.isdigit() and 0 <= int(value) <= total_clusters
        ))

        # Identify the largest clusters and prompt for confirmation
        clusters_to_display = find_largest_clusters(cluster_labels, num_clusters_to_display)
        confirmed_clusters = confirm_choice(
            f"You selected {num_clusters_to_display} clusters to display. Is that correct?",
            clusters_to_display
        )

        # If confirmed, display the cluster details; if not, re-prompt
        if confirmed_clusters:
            clusters = create_cluster_dict(cluster_labels, metadata, confirmed_clusters)
            display_cluster_details(clusters)
            return clusters

    