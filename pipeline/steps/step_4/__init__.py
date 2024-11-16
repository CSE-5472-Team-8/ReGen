import numpy as np
import os
from sklearn.cluster import DBSCAN
from steps.settings import find_clusters_settings

def load_embeddings(embedding_dir):
    """
    Load all embedding files from a directory. Extract embeddings and metadata from .npz files.
    
    Args:
        embedding_dir (str): Path to the directory containing embedding files.
    
    Returns:
        tuple: 
            - A NumPy array containing all embeddings.
            - A list of dictionaries with metadata (e.g., url, index, caption).
    """
    print("Reading embeddings...")
    embeddings_metadata = []
    embeddings = []
    for filename in os.listdir(embedding_dir):
        if filename.endswith('.npz'):
            data = np.load(os.path.join(embedding_dir, filename))
            embeddings.append(data['embedding'])
            
            index = data['index'].item() if 'index' in data and isinstance(data['index'], np.ndarray) else data.get('index')
            caption = data['caption'].item() if 'caption' in data and isinstance(data['caption'], np.ndarray) else data.get('caption')
            
            embeddings_metadata.append({
                'url': data['url'] if 'url' in data else None,
                'index': index,
                'caption': caption
            })

    return np.array(embeddings).squeeze(), embeddings_metadata

def validate_embeddings_shape(embeddings):
    """
    Validate the shape of the embeddings array to ensure compatibility with clustering.
    
    Args:
        embeddings (np.ndarray): The array of embeddings to validate.
    
    Raises:
        ValueError: If the embeddings do not have the required shape (n_samples, 512).
    """
    if embeddings.shape[1] != 512:
        raise ValueError(f"Embeddings should have shape (n_samples, 512). Current shape is {embeddings.shape}")

def cluster_embeddings(embeddings, eps=0.5, min_samples=2):
    """
    Perform clustering on the embeddings using the DBSCAN algorithm.
    
    Args:
        embeddings (np.ndarray): Array of embeddings to cluster.
        eps (float): The maximum distance between two samples for one to be considered in the neighborhood of the other.
        min_samples (int): The number of samples required to form a cluster.
    
    Returns:
        np.ndarray: An array of cluster labels for each embedding.
    """
    print("Finding clusters...")

    eps = find_clusters_settings['epsilon']
    min_samples = find_clusters_settings['min_samples_per_cluster']
    metric = find_clusters_settings['clustering_metric']

    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    cluster_labels = clustering.fit_predict(embeddings)
    return cluster_labels

def find_clusters(embedding_dir):
    """
    Execute the entire process of loading embeddings, validating their shape, and clustering them.
    
    Args:
        embedding_dir (str): Path to the directory containing embedding files.
    
    Returns:
        tuple: 
            - An array of cluster labels for each embedding.
            - A list of dictionaries containing metadata for each embedding.
    """
    embeddings, embeddings_metadata = load_embeddings(embedding_dir)
    validate_embeddings_shape(embeddings)
    cluster_labels = cluster_embeddings(embeddings)
    return cluster_labels, embeddings_metadata
