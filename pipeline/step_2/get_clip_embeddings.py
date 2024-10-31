import torch
import clip
from PIL import Image
import numpy as np
import requests
from io import BytesIO
from tqdm import tqdm
import os
from datasets import load_dataset
from huggingface_hub import login
from dotenv import load_dotenv
from step_2 import get_dataset_size_rows

class ImageEmbedder:
    def __init__(self, dataset_id, embedding_dir, feature_names, max_batch_size_gb=2):
        """
        Initialize the ImageEmbedder with dataset information, device setup, and model loading.

        Args:
            dataset_id (str): The dataset identifier for loading from Hugging Face.
            embedding_dir (str): Directory path for saving the embeddings.
            feature_names (dict): Feature names to identify images, captions, and URLs in the dataset.
            max_batch_size_gb (int, optional): Maximum batch size in gigabytes. Defaults to 2.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.dataset = load_dataset(dataset_id, split="train", streaming=True)
        self.dataset_num_rows = get_dataset_size_rows.run(dataset_id)
        self.embedding_dir = embedding_dir
        self.feature_names = feature_names
        self.max_batch_size_bytes = max_batch_size_gb * 1024**3

    def download_and_embed_image(self, image, image_url, caption, embedding_save_path, index):
        """
        Download an image from a URL if not provided directly, preprocess it, 
        generate CLIP embeddings, and save them along with metadata.

        Args:
            image (PIL.Image or None): Image object or None if only URL is provided.
            image_url (str or None): URL of the image if not directly provided.
            caption (str): Caption associated with the image.
            embedding_save_path (str): Path to save the embedding file.
            index (int): Index of the image in the dataset.

        Returns:
            int: Size in bytes of the generated embedding, or 0 if embedding generation fails.
        """
        try:
            if image is None and image_url is not None:
                response = requests.get(image_url, timeout=3)
                if response.status_code == 200:
                    image = Image.open(BytesIO(response.content))
                else:
                    return 0
            elif image is None:
                return 0

            image = self.preprocess(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                image_features = self.model.encode_image(image).cpu().numpy()

            if image_url:
                np.savez(embedding_save_path, embedding=image_features, url=image_url, caption=caption)
            else:
                np.savez(embedding_save_path, embedding=image_features, index=index, caption=caption)
            
            return image_features.nbytes
        except requests.exceptions.Timeout:
            return 0
        except Exception:
            return 0

    def process_images_in_batches(self, batch_size=1000, total_samples=1_000_000):
        """
        Process images in batches, download and embed them, and save embeddings until the limit is reached.

        Args:
            batch_size (int, optional): Number of samples to process in a single batch. Defaults to 1000.
            total_samples (int, optional): Total number of samples to process. Defaults to 1,000,000.
        """
        os.makedirs(self.embedding_dir, exist_ok=True)

        load_dotenv()
        login(token=os.getenv('HUGGINGFACE_TOKEN'))

        batch = []
        processed_samples = 0
        batch_size_bytes = 0

        scenario = self.determine_scenario()

        for example_index, example in enumerate(tqdm(self.dataset, total=total_samples)):
            if scenario == 1 and self.feature_names["image"] in example and self.feature_names["caption"] in example:
                image = example[self.feature_names["image"]]
                caption = example[self.feature_names["caption"]]
                batch.append({"image": image, "caption": caption, "index": example_index})
            elif scenario == 2 and all(key in example for key in [self.feature_names["image"], self.feature_names["image_url"], self.feature_names["caption"]]):
                image = example[self.feature_names["image"]]
                image_url = example[self.feature_names["image_url"]]
                caption = example[self.feature_names["caption"]]
                batch.append({"image": image, "image_url": image_url, "caption": caption, "index": example_index})
            elif scenario == 3 and self.feature_names["image_url"] in example and self.feature_names["caption"] in example:
                image_url = example[self.feature_names["image_url"]]
                caption = example[self.feature_names["caption"]]
                batch.append({"image": None, "image_url": image_url, "caption": caption, "index": example_index})

            if len(batch) == batch_size or batch_size_bytes >= self.max_batch_size_bytes:
                for sample in batch:
                    image = sample.get("image")
                    image_url = sample.get("image_url")
                    caption = sample.get("caption", 'No caption available')
                    index = sample.get("index")
                    embedding_save_path = os.path.join(self.embedding_dir, f"embedding_{processed_samples}.npz")

                    size = self.download_and_embed_image(image, image_url, caption, embedding_save_path, index)
                    if size > 0:
                        processed_samples += 1
                        batch_size_bytes += size

                batch = []
                batch_size_bytes = 0

            if processed_samples >= total_samples:
                break

        if batch:
            for sample in batch:
                image = sample.get("image")
                image_url = sample.get("image_url")
                caption = sample.get("caption", 'No caption available')
                index = sample.get("index")
                embedding_save_path = os.path.join(self.embedding_dir, f"embedding_{processed_samples}.npz")
                size = self.download_and_embed_image(image, image_url, caption, embedding_save_path, index)
                if size > 0:
                    processed_samples += 1

    def determine_scenario(self):
        """
        Determine the embedding scenario based on the provided feature names.

        Returns:
            int: Scenario number (1, 2, or 3) based on available features in the dataset.
        """
        keys = set(self.feature_names.keys())
        if keys == {"image", "caption"}:
            return 1
        elif keys == {"image", "image_url", "caption"}:
            return 2
        elif keys == {"image_url", "caption"}:
            return 3
        else:
            raise ValueError("Invalid combination of feature names.")

def run(dataset_id, feature_names):
    """
    Run the image embedding pipeline on a specified dataset with given feature names.

    Args:
        dataset_id (str): The dataset identifier for loading.
        feature_names (dict): Feature names for identifying image, caption, and image URL in the dataset.
    """
    embedding_dir = f"./data/clip_embeddings/{dataset_id.replace('/', '_')}"
    embedder = ImageEmbedder(dataset_id, embedding_dir, feature_names)
    
    embedder.process_images_in_batches(batch_size=1000, total_samples=embedder.dataset_num_rows)
