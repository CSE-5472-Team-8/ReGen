import os
from CLI.step_1 import select_model_and_dataset
from CLI.step_2 import setup_dataset
from step_3 import get_clip_embeddings
from step_4 import find_clusters
from CLI.step_5 import select_clusters_to_attack
from step_6 import get_images_from_dataset

def main():
    """Run the model inversion attack pipeline."""
    
    # Step 1: Select model and dataset
    model_id, dataset_id = select_model_and_dataset()
    
    # Step 2: Get information about the dataset from the user.
    feature_names = setup_dataset(dataset_id)

    # Step 3: Generate CLIP embeddings for all images in the dataset.
    embedding_dir = get_clip_embeddings.run(dataset_id, feature_names)
    
    # Step 4: Identify the most duplicated images in the dataset.
    cluster_labels, embeddings_metadata = find_clusters.run(embedding_dir)
    
    # Step 5: Decide which duplicated images to target for the model inversion attack.
    clusters = select_clusters_to_attack(cluster_labels, embeddings_metadata)
    
    # Step 6: Using the selected model, generate X images using the caption associated with the most duplicated images.
    get_images_from_dataset.run(dataset_id, clusters, feature_names)

    # Step 7: Compare duplicated images with generated images to identify instances of memorized training data.

if __name__ == "__main__":
    main()
