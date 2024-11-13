import numpy as np
import os
from sklearn.cluster import DBSCAN
from steps.settings import find_clusters_settings

# Load all embeddings into a list of dictionaries and separate embedding arrays
def load_embeddings(embedding_dir):
    print("Reading embeddings...")
    embeddings_metadata = []
    embeddings = []
    for filename in os.listdir(embedding_dir):
        if filename.endswith('.npz'):
            data = np.load(os.path.join(embedding_dir, filename))
            embeddings.append(data['embedding'])
            
            # Extract metadata, ensuring single-element arrays are converted to scalars
            index = data['index'].item() if 'index' in data and isinstance(data['index'], np.ndarray) else data.get('index')
            caption = data['caption'].item() if 'caption' in data and isinstance(data['caption'], np.ndarray) else data.get('caption')
            
            embeddings_metadata.append({
                'url': data['url'] if 'url' in data else None,
                'index': index,
                'caption': caption
            })

    return np.array(embeddings).squeeze(), embeddings_metadata

# Validate embeddings
def validate_embeddings_shape(embeddings):
    if embeddings.shape[1] != 512:
        raise ValueError(f"Embeddings should have shape (n_samples, 512). Current shape is {embeddings.shape}")

# Cluster embeddings using DBSCAN
def cluster_embeddings(embeddings, eps=0.5, min_samples=2):
    print("Finding clusters...")

    eps = find_clusters_settings['epsilon']
    min_samples = find_clusters_settings['min_samples_per_cluster']
    metric = find_clusters_settings['clustering_metric']

    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    cluster_labels = clustering.fit_predict(embeddings)
    return cluster_labels

# Main run function to execute all steps
def run(embedding_dir):
    embeddings, embeddings_metadata = load_embeddings(embedding_dir)  # Get both data and array
    validate_embeddings_shape(embeddings)
    cluster_labels = cluster_embeddings(embeddings)
    return cluster_labels, embeddings_metadata
