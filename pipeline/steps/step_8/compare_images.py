from tqdm import tqdm
import os
import cv2
import numpy as np
from pathlib import Path
from steps.step_8.image_similarity_measures import evaluation
from steps.settings import image_comparison_settings
import warnings

weights = image_comparison_settings['metric_weights']
memorization_likelihood_threshold = image_comparison_settings['memorization_likelihood_threshold']

def calculate_memorization_likelihood(normalized_scores):
    likelihood_score = sum(normalized_scores[metric] * weights[metric] for metric in normalized_scores.keys())
    is_memorized = likelihood_score > memorization_likelihood_threshold
    
    return likelihood_score, is_memorized

def normalize_scores(scores_dict):
    ranges = {
        "fsim": (0, 1),
        "psnr": (0, 50),
        "rmse": (0, 1),
        "sre": (0, 60),
        "ssim": (0, 1),
        "uiq": (0, 1)
    }
    
    normalized_metrics = {}
    for metric, (min_val, max_val) in ranges.items():
        normalized_values = [(score - min_val) / (max_val - min_val) for score in scores_dict[metric]]
        if metric in ["rmse", "sam"]:
            normalized_values = [1 - val for val in normalized_values]
        
        normalized_metrics[metric] = normalized_values

    return normalized_metrics

def compare_images(input_image_path, image_paths):
    metric_names = ["fsim", "psnr", "rmse", "sre", "ssim", "uiq"]
    raw_scores = {metric: [] for metric in metric_names}

    input_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    target_size = (input_image.shape[1], input_image.shape[0])

    print()
    for comparison_image_path in tqdm(image_paths, desc="Comparing images"):
        comparison_image = cv2.imread(comparison_image_path, cv2.IMREAD_GRAYSCALE)
        
        if comparison_image is None:
            raise ValueError(f"Unable to read image at {comparison_image_path}")

        resized_input_image = cv2.resize(input_image, target_size)
        resized_comparison_image = cv2.resize(comparison_image, target_size)

        cv2.imwrite("temp_resized_input_image.png", resized_input_image)
        cv2.imwrite("temp_comparison_image.png", resized_comparison_image)
        
        # Suppress warnings during the evaluation function call
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            score = evaluation(org_img_path="temp_resized_input_image.png", pred_img_path="temp_comparison_image.png", metrics=metric_names)
        
        # Clean up temporary files
        os.remove("temp_resized_input_image.png")
        os.remove("temp_comparison_image.png")

        for metric in metric_names:
            raw_scores[metric].append(score[metric])

    normalized_scores = normalize_scores(raw_scores)
    combined_scores = [sum([normalized_scores[metric][i] for metric in metric_names]) for i in range(len(image_paths))]
    most_similar_idx = np.argmax(combined_scores)

    return most_similar_idx, normalized_scores


def run(model_id, dataset_id, clusters):
    for cluster_id in clusters.keys():
        original_image_path = f"./data/clusters/{dataset_id.replace('/', '_')}/cluster_{cluster_id}.png"
        generated_images_path = f"./data/generated_images/{model_id.replace('/', '_')}/cluster_{cluster_id}"

        generated_image_paths = list(Path(generated_images_path).glob("*"))
        generated_image_paths = [str(p) for p in generated_image_paths if p.is_file()]

        if not generated_image_paths:
            raise ValueError(f"No valid image files found in {generated_images_paths}")

        most_similar_index, normalized_scores = compare_images(original_image_path, generated_image_paths)

        most_similar_path = generated_image_paths[most_similar_index]
        print(f"\nMost similar image: {most_similar_path}")

        normalized_scores_for_most_similar = {metric: values[most_similar_index] for metric, values in normalized_scores.items()}
        likelihood_score, is_memorized = calculate_memorization_likelihood(normalized_scores_for_most_similar)

        print(f"Memorization likelihood score (out of 1): {likelihood_score}")
        print(f"Is the image likely memorized? {'Yes' if is_memorized else 'No'}")
