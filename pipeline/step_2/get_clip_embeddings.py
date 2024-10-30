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
    def __init__(self, dataset_id, embedding_dir):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.dataset = load_dataset(dataset_id, split="train", streaming=True)
        self.embedding_dir = embedding_dir

    def download_and_embed_image(self, image_url, caption, embedding_save_path, timeout=3):
        try:
            response = requests.get(image_url, timeout=timeout)
            if response.status_code == 200:
                image = Image.open(BytesIO(response.content))
                image = self.preprocess(image).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    image_features = self.model.encode_image(image).cpu().numpy()

                np.savez(embedding_save_path, embedding=image_features, url=image_url, caption=caption)
                return True
            else:
                return False
        except requests.exceptions.Timeout:
            return False
        except Exception:
            return False

    def process_images_in_batches(self, batch_size=1000, total_samples=1_000_000, max_workers=10):
        os.makedirs(self.embedding_dir, exist_ok=True)

        load_dotenv()
        login(token=os.getenv('HUGGINGFACE_TOKEN'))

        batch = []
        processed_samples = 0
        futures = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for example in tqdm(self.dataset, total=total_samples):
                if 'url' in example and 'caption' in example:
                    batch.append(example)

                if len(batch) == batch_size:
                    for sample in batch:
                        image_url = sample['url']
                        caption = sample.get('caption', 'No caption available')
                        embedding_save_path = os.path.join(self.embedding_dir, f"embedding_{processed_samples}.npz")

                        futures.append(
                            executor.submit(self.download_and_embed_image, image_url, caption, embedding_save_path)
                        )

                    for future in as_completed(futures):
                        if future.result():
                            processed_samples += 1

                    batch = []
                    futures = []

                if processed_samples >= total_samples:
                    break

            if batch:
                for sample in batch:
                    image_url = sample['url']
                    caption = sample.get('caption', 'No caption available')
                    embedding_save_path = os.path.join(self.embedding_dir, f"embedding_{processed_samples}.npz")
                    futures.append(
                        executor.submit(self.download_and_embed_image, image_url, caption, embedding_save_path)
                    )

                for future in as_completed(futures):
                    if future.result():
                        processed_samples += 1

def run(dataset_id):
    embedding_dir = f"./data/clip_embeddings/{dataset_id.replace('/', '_')}"
    embedder = ImageEmbedder(dataset_id, embedding_dir)
    
    embedder.process_images_in_batches(batch_size=10, total_samples=30, max_workers=2)

