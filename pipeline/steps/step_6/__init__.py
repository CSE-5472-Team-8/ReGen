import os
import requests
import hashlib
from PIL import Image
from io import BytesIO
from datasets import load_dataset

def calculate_image_hash(image_data):
    """
    Calculate the SHA-256 hash of the given image data.
    
    Args:
        image_data (bytes): Binary data of the image.
    
    Returns:
        str: The SHA-256 hash of the image data as a hexadecimal string.
    """
    return hashlib.sha256(image_data).hexdigest()

def get_images_from_dataset(dataset_id, clusters, feature_names):
    """
    Retrieve images from a dataset, either via URLs or from dataset features, and save unique images to disk.
    
    Args:
        dataset_id (str): Identifier of the dataset to load.
        clusters (dict): Dictionary mapping cluster IDs to lists of dataset items.
        feature_names (dict): Dictionary mapping feature types (e.g., 'image') to dataset feature keys.
    
    Returns:
        None
    """
    image_count = 0
    saved_image_hashes = set()
    dataset = load_dataset(dataset_id)

    for cluster_id, items in clusters.items():
        cluster_dir = f"./data/clusters/{dataset_id.replace('/', '_')}"
        os.makedirs(cluster_dir, exist_ok=True)

        for item in items:
            image_filename = f"{cluster_dir}/cluster_{cluster_id}.png"

            if 'url' in item and item['url'] is not None:
                # Get the image from its URL
                image_url = item['url']
                response = requests.get(image_url, timeout=3)
                if response.status_code == 200:
                    image_data = response.content
                    image_hash = calculate_image_hash(image_data)
            else:
                # Get the image directly from the dataset
                row_index = item['index']
                data = dataset['train'][row_index]
                image = data[feature_names['image']]
                buffered = BytesIO()
                image.save(buffered, format="PNG")
                image_data = buffered.getvalue()
                image_hash = calculate_image_hash(image_data)

            # Save the image if it hasn't been saved yet
            if image_data and image_hash not in saved_image_hashes:
                with open(image_filename, "wb") as f:
                    f.write(image_data)
                saved_image_hashes.add(image_hash)
                image_count += 1

    print(f"Total number of unique images saved: {image_count}")
