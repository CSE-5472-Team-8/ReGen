from tqdm import tqdm
import os
import cv2
import numpy as np
from pathlib import Path
from steps.step_8.image_similarity_measures import evaluation
from steps.settings import image_comparison_settings
import warnings

# Global configuration
weights = image_comparison_settings['metric_weights']
memorization_likelihood_threshold = image_comparison_settings['memorization_likelihood_threshold']
metric_names = ["fsim", "psnr", "rmse", "sre", "ssim", "uiq"]
normalization_ranges = {
    "fsim": (0, 1),
    "psnr": (0, 50),
    "rmse": (0, 1),
    "sre": (0, 60),
    "ssim": (0, 1),
    "uiq": (0, 1)
}


def calculate_memorization_likelihood(normalized_scores):
    """
    Calculate the likelihood of memorization based on normalized similarity scores and pre-defined weights.
    
    Args:
        normalized_scores (dict): Dictionary of normalized scores for each metric.
    
    Returns:
        tuple: Memorization likelihood score and a boolean indicating whether the image is likely memorized.
    """
    likelihood_score = sum(normalized_scores[metric] * weights[metric] for metric in normalized_scores)
    is_memorized = likelihood_score > memorization_likelihood_threshold
    return likelihood_score, is_memorized


def normalize_scores(scores_dict):
    """
    Normalize similarity scores based on metric-specific ranges.
    
    Args:
        scores_dict (dict): Raw scores for each metric.
    
    Returns:
        dict: Normalized scores for each metric.
    """
    normalized_metrics = {}
    for metric, (min_val, max_val) in normalization_ranges.items():
        normalized_values = [(score - min_val) / (max_val - min_val) for score in scores_dict[metric]]
        if metric in ["rmse", "sam"]:
            normalized_values = [1 - val for val in normalized_values]
        normalized_metrics[metric] = normalized_values
    return normalized_metrics


def process_and_compare_images(input_image_path, comparison_image_path):
    """
    Resize and compare two images using similarity metrics.
    
    Args:
        input_image_path (str): Path to the input (original) image.
        comparison_image_path (str): Path to the comparison image.
    
    Returns:
        dict: Similarity scores for all metrics.
    """
    input_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    comparison_image = cv2.imread(comparison_image_path, cv2.IMREAD_GRAYSCALE)

    if input_image is None or comparison_image is None:
        raise ValueError(f"Unable to read image(s): {input_image_path}, {comparison_image_path}")

    target_size = (input_image.shape[1], input_image.shape[0])
    resized_input_image = cv2.resize(input_image, target_size)
    resized_comparison_image = cv2.resize(comparison_image, target_size)

    temp_input_path = "temp_resized_input_image.png"
    temp_comparison_path = "temp_resized_comparison_image.png"
    cv2.imwrite(temp_input_path, resized_input_image)
    cv2.imwrite(temp_comparison_path, resized_comparison_image)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = evaluation(org_img_path=temp_input_path, pred_img_path=temp_comparison_path, metrics=metric_names)

    os.remove(temp_input_path)
    os.remove(temp_comparison_path)

    return scores


def compare_generated_and_training_images(input_image_path, image_paths):
    """
    Compare a training image against multiple generated images and identify the most similar one.
    
    Args:
        input_image_path (str): Path to the original training image.
        image_paths (list): List of paths to generated images.
    
    Returns:
        tuple: Index of the most similar image and normalized scores for all metrics.
    """
    raw_scores = {metric: [] for metric in metric_names}

    for comparison_image_path in tqdm(image_paths, desc="Comparing images"):
        scores = process_and_compare_images(input_image_path, comparison_image_path)
        for metric in metric_names:
            raw_scores[metric].append(scores[metric])

    normalized_scores = normalize_scores(raw_scores)
    combined_scores = [sum(normalized_scores[metric][i] for metric in metric_names) for i in range(len(image_paths))]
    most_similar_idx = np.argmax(combined_scores)

    return most_similar_idx, normalized_scores


def compare_images(model_id, dataset_id, clusters):
    """
    Compare generated images to training images for each cluster and assess memorization likelihood.
    
    Args:
        model_id (str): Model identifier for the generated images.
        dataset_id (str): Dataset identifier for the training images.
        clusters (dict): Dictionary of clusters with associated data.
    
    Returns:
        None
    """
    for cluster_id in clusters.keys():
        original_image_path = f"./data/clusters/{dataset_id.replace('/', '_')}/cluster_{cluster_id}.png"
        generated_images_path = f"./data/generated_images/{model_id.replace('/', '_')}/cluster_{cluster_id}"

        generated_image_paths = list(Path(generated_images_path).glob("*"))
        generated_image_paths = [str(p) for p in generated_image_paths if p.is_file()]

        if not generated_image_paths:
            raise ValueError(f"No valid image files found in {generated_images_path}")

        most_similar_index, normalized_scores = compare_generated_and_training_images(original_image_path, generated_image_paths)

        most_similar_path = generated_image_paths[most_similar_index]
        print(f"\nMost similar image: {most_similar_path}")

        normalized_scores_for_most_similar = {metric: values[most_similar_index] for metric, values in normalized_scores.items()}
        likelihood_score, is_memorized = calculate_memorization_likelihood(normalized_scores_for_most_similar)

        print(f"Memorization likelihood score (out of 1): {likelihood_score}")
        print(f"Is the image likely memorized? {'Yes' if is_memorized else 'No'}")
