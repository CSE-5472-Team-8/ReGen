import select_model_and_dataset
import get_clip_embeddings

if __name__ == "__main__":
    # Step 1: Choose the model you wish to run the model inversion attack on, and the dataset it was trained on.
    model_id, dataset_id = select_model_and_dataset.run()

    # Step 2: Generate CLIP embeddings for all images in the dataset.
    get_clip_embeddings.run(dataset_id)

    # Step 3: Identify the most duplicated images in the dataset.

    # Step 4: Using the selected model, generate X images using the caption associated with the most duplicated images.

    # Step 5: Compare duplicated images with generated images to identify instances of memorized training data.