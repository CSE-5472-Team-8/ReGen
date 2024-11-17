import os
from diffusers import StableDiffusionPipeline, DiffusionPipeline, DPMSolverMultistepScheduler
from huggingface_hub import model_info
import torch
from steps.settings import image_generation_settings


def generate_images(model_id, clusters):
    """
    Generate images from a text-to-image model on Hugging Face using the provided prompts.
    
    Args:
        model_id (str): The model ID on Hugging Face.
        clusters (dict): A dictionary where each key is a cluster ID and the value is a list of items with prompts.
    
    Returns:
        None
    """
    generated_images_dir = create_generated_images_directory(model_id)

    if check_existing_images(generated_images_dir, clusters):
        print("\nAll clusters already have the required images. Skipping generation.")
        return

    pipeline = initialize_pipeline(model_id)
    if not pipeline:
        print("\nModel not supported or incompatible.")
        return

    generate_images_for_clusters(pipeline, generated_images_dir, clusters)


def create_generated_images_directory(model_id):
    """
    Create and return the path to the directory where generated images will be saved.
    """
    directory = f"./data/generated_images/{model_id.replace('/', '_')}"
    os.makedirs(directory, exist_ok=True)
    return directory


def check_existing_images(base_dir, clusters):
    """
    Check if all clusters already have the required number of generated images.
    
    Args:
        base_dir (str): Base directory for generated images.
        clusters (dict): Dictionary of clusters.
    
    Returns:
        bool: True if all clusters are complete, False otherwise.
    """
    for cluster_id, items in clusters.items():
        cluster_dir = os.path.join(base_dir, f"cluster_{cluster_id}")
        os.makedirs(cluster_dir, exist_ok=True)

        existing_images = len([f for f in os.listdir(cluster_dir) if f.endswith(".png")])
        if existing_images < image_generation_settings['num_generated_images_per_cluster']:
            return False
    return True


def initialize_pipeline(model_id):
    """
    Initialize and configure the appropriate DiffusionPipeline or StableDiffusionPipeline.
    
    Args:
        model_id (str): The model ID on Hugging Face.
    
    Returns:
        DiffusionPipeline: Configured pipeline, or None if the model is unsupported.
    """
    card_data = model_info(model_id).card_data.to_dict()
    tags = card_data.get('tags', [])

    if 'diffusers' not in tags:
        return None

    if 'base_model' in card_data:
        base_model_id = card_data['base_model']
        print(f"\nLoading model {base_model_id} with LoRA weights from {model_id}...")
        pipeline = DiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16)
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)

        if base_model_id != model_id:
            pipeline.load_lora_weights(model_id)
    else:
        print(f"\nLoading model {model_id}...")
        pipeline = StableDiffusionPipeline.from_pretrained(model_id)

    pipeline.to("cuda" if torch.cuda.is_available() else "cpu")
    return pipeline


def generate_images_for_clusters(pipeline, base_dir, clusters):
    """
    Generate images for clusters that require additional images.
    
    Args:
        pipeline (DiffusionPipeline): Initialized pipeline for image generation.
        base_dir (str): Base directory for saving images.
        clusters (dict): Dictionary of clusters with prompts.
    """
    num_images_per_cluster = image_generation_settings['num_generated_images_per_cluster']
    num_inference_steps = image_generation_settings['num_inference_steps']

    for cluster_id, items in clusters.items():
        cluster_dir = os.path.join(base_dir, f"cluster_{cluster_id}")
        existing_images = len([f for f in os.listdir(cluster_dir) if f.endswith(".png")])

        if existing_images >= num_images_per_cluster:
            print(f"\nCluster {cluster_id} already has {existing_images} images. Skipping generation.")
            continue

        os.makedirs(cluster_dir, exist_ok=True)
        print(f"\nGenerating images for Cluster {cluster_id}...")

        for i in range(num_images_per_cluster):
            item = items[i % len(items)]
            prompt = item['caption']
            image_save_path = os.path.join(cluster_dir, f"generated_image_{i}.png")

            print(f"\nGenerating image for prompt: '{prompt}'")
            image = pipeline(prompt, num_inference_steps=num_inference_steps).images[0]
            image.save(image_save_path)
            print(f"Image saved to {image_save_path}")
