import os
from diffusers import StableDiffusionPipeline, DiffusionPipeline, DPMSolverMultistepScheduler
from huggingface_hub import model_info
import torch
from steps.settings import image_generation_settings

def run(model_id, clusters):
    """
    Generate images from a text-to-image model on Hugging Face using the provided prompts.
    
    Args:
    - model_id (str): The model ID on Hugging Face.
    - clusters (dict): A dictionary where each key is a cluster ID and the value is a list of items with prompts.
    
    Returns:
    - None
    """
    generated_images_dir = f"./data/generated_images/{model_id.replace('/', '_')}"
    os.makedirs(generated_images_dir, exist_ok=True)

    # Check if all clusters have the required images before loading the model
    all_clusters_complete = True
    for cluster_id, items in clusters.items():
        cluster_dir = os.path.join(generated_images_dir, f"cluster_{cluster_id}")
        os.makedirs(cluster_dir, exist_ok=True)

        # Check the number of existing images
        existing_images = len([f for f in os.listdir(cluster_dir) if f.endswith(".png")])
        num_images_required = image_generation_settings['num_generated_images_per_cluster']
        
        if existing_images < num_images_required:
            all_clusters_complete = False
            break

    # Exit early if all clusters already have enough images
    if all_clusters_complete:
        print("\nAll clusters already have the required images. Skipping generation.")
        return

    # Load model metadata and check if it's compatible
    card_data = model_info(model_id).card_data.to_dict()
    tags = card_data['tags']

    if 'diffusers' not in tags:
        print('\nModel not supported.')
        return

    # Load the model and configure it
    pipeline = None
    if 'base_model' in card_data.keys():
        base_model_id = card_data['base_model']
        
        print(f"\nLoading model {base_model_id} with LoRA weights from {model_id}...")
        pipeline = DiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16)
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)

        if base_model_id != model_id:
            pipeline.load_lora_weights(model_id)
        
        pipeline.to("cuda" if torch.cuda.is_available() else "cpu")

    else:
        pipeline = StableDiffusionPipeline.from_pretrained(model_id)

    pipeline.to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate images for clusters that still need them
    for cluster_id, items in clusters.items():
        cluster_dir = os.path.join(generated_images_dir, f"cluster_{cluster_id}")
        existing_images = len([f for f in os.listdir(cluster_dir) if f.endswith(".png")])
        num_images_required = image_generation_settings['num_generated_images_per_cluster']
        
        if existing_images >= num_images_required:
            print(f"\nCluster {cluster_id} already has {existing_images} images. Skipping generation.")
            continue

        # Generate images if needed
        for i in range(num_images_required):
            item = items[i % len(items)]
            image_save_path = os.path.join(cluster_dir, f"generated_image_{i}.png")
            
            prompt = item['caption']
            print(f"\nGenerating image for prompt: '{prompt}'")
            image = pipeline(prompt, num_inference_steps=image_generation_settings['num_inference_steps']).images[0]
            image.save(image_save_path)
            print(f"Image saved to {image_save_path}")
