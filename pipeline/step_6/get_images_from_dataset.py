import os
from datasets import load_dataset

def run(dataset_id, clusters, feature_names):
    dataset = load_dataset(dataset_id)

    for cluster_id, items in clusters.items():
        cluster_dir = f"./data/clusters/{dataset_id.replace('/', '_')}/cluster_{cluster_id}"
        os.makedirs(cluster_dir, exist_ok=True)

        for item in items:
            row_index = item['index']
            data = dataset['train'][row_index]

            image = data[feature_names['image']]
            caption = data[feature_names['caption']]

            image_filename = f"{cluster_dir}/image_{row_index}.png"

            image.save(image_filename)
            print(f"Image with caption '{caption}' saved to: {image_filename}")

