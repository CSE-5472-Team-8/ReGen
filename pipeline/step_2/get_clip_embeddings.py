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
from concurrent.futures import ThreadPoolExecutor, as_completed

class ImageEmbedder:
    def __init__(self, dataset_id, embedding_dir, feature_names, max_batch_size_gb=2):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.dataset = load_dataset(dataset_id, split="train", streaming=True)
        self.embedding_dir = embedding_dir
        self.feature_names = feature_names
        self.max_batch_size_bytes = max_batch_size_gb * 1024**3

    def download_and_embed_image(self, image_url, caption, embedding_save_path, timeout=3):
        try:
            response = requests.get(image_url, timeout=timeout)
            if response.status_code == 200:
                image = Image.open(BytesIO(response.content))
                image = self.preprocess(image).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    image_features = self.model.encode_image(image).cpu().numpy()

                np.savez(embedding_save_path, embedding=image_features, url=image_url, caption=caption)
                return len(response.content) + image_features.nbytes
            else:
                return 0
        except requests.exceptions.Timeout:
            return 0
        except Exception:
            return 0

    def process_images_in_batches(self, batch_size=1000, total_samples=1_000_000, max_workers=10):
        os.makedirs(self.embedding_dir, exist_ok=True)

        load_dotenv()
        login(token=os.getenv('HUGGINGFACE_TOKEN'))

        batch = []
        processed_samples = 0
        batch_size_bytes = 0
        futures = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for example in tqdm(self.dataset, total=total_samples):
                if 'url' in example and 'caption' in example:
                    batch.append(example)

                if len(batch) == batch_size or batch_size_bytes >= self.max_batch_size_bytes:
                    for sample in batch:
                        image_url = sample['url']
                        caption = sample.get('caption', 'No caption available')
                        embedding_save_path = os.path.join(self.embedding_dir, f"embedding_{processed_samples}.npz")

                        future = executor.submit(self.download_and_embed_image, image_url, caption, embedding_save_path)
                        futures.append(future)

                    for future in as_completed(futures):
                        size = future.result()
                        if size > 0:
                            processed_samples += 1
                            batch_size_bytes += size

                    batch = []
                    batch_size_bytes = 0
                    futures = []

                if processed_samples >= total_samples:
                    break

            if batch:
                for sample in batch:
                    image_url = sample['url']
                    caption = sample.get('caption', 'No caption available')
                    embedding_save_path = os.path.join(self.embedding_dir, f"embedding_{processed_samples}.npz")
                    future = executor.submit(self.download_and_embed_image, image_url, caption, embedding_save_path)
                    futures.append(future)

                for future in as_completed(futures):
                    size = future.result()
                    if size > 0:
                        processed_samples += 1

def run(dataset_id, feature_names):
    embedding_dir = f"./data/clip_embeddings/{dataset_id.replace('/', '_')}"
    embedder = ImageEmbedder(dataset_id, embedding_dir, feature_names)
    
    embedder.process_images_in_batches(batch_size=10, total_samples=30, max_workers=2)
