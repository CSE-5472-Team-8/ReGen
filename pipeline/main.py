from CLI.step_1 import select_model_and_dataset
from CLI.step_2 import setup_dataset
from step_3 import get_clip_embeddings
from step_4 import find_clusters

if __name__ == "__main__":
    # Step 1: Choose the model you wish to run the model inversion attack on, and the dataset it was trained on.
    model_id, dataset_id = select_model_and_dataset()

    # Step 2: Get information about the dataset from the user.
    feature_names = setup_dataset(dataset_id)

    # Step 3: Generate CLIP embeddings for all images in the dataset.
    embedding_dir = get_clip_embeddings.run(dataset_id, feature_names)

    # Step 4: Identify the most duplicated images in the dataset.
    find_clusters.run(embedding_dir)

    # Step 5: Using the selected model, generate X images using the caption associated with the most duplicated images.

    # Step 6: Compare duplicated images with generated images to identify instances of memorized training data.