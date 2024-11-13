import os
import requests
import hashlib
from PIL import Image
from io import BytesIO
from datasets import load_dataset

def calculate_image_hash(image_data):
    """Calculate the SHA-256 hash of the image data."""
    return hashlib.sha256(image_data).hexdigest()

def run(dataset_id, clusters, feature_names):
    image_count = 0
    saved_hashes = set()  # To store hashes of saved images
    dataset = load_dataset(dataset_id)

    for cluster_id, items in clusters.items():
        cluster_dir = f"./data/clusters/{dataset_id.replace('/', '_')}"
        os.makedirs(cluster_dir, exist_ok=True)

        for item in items:
            row_index = item['index']
            data = dataset['train'][row_index]
            image_filename = f"{cluster_dir}/cluster_{cluster_id}.png"

            # Check if there's an image URL or an image feature in the dataset
            if 'url' in item and item['url'] is not None:
                image_url = item['url']
                response = requests.get(image_url, timeout=3)
                if response.status_code == 200:
                    image_data = response.content
                    image_hash = calculate_image_hash(image_data)

                    # Only save if the hash is not in saved_hashes
                    if image_hash not in saved_hashes:
                        with open(image_filename, "wb") as f:
                            f.write(image_data)
                        saved_hashes.add(image_hash)
                        image_count += 1

            else:
                # Save the image from the dataset feature
                image = data[feature_names['image']]
                buffered = BytesIO()
                image.save(buffered, format="PNG")
                image_data = buffered.getvalue()
                image_hash = calculate_image_hash(image_data)

                # Only save if the hash is not in saved_hashes
                if image_hash not in saved_hashes:
                    with open(image_filename, "wb") as f:
                        f.write(image_data)
                    saved_hashes.add(image_hash)
                    image_count += 1

    print(f"Total number of unique images saved: {image_count}")
