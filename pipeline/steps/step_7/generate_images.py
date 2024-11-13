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
    - prompts (list): A list of prompts (strings) for image generation.
    
    Returns:
    - images (list): A list of PIL Image objects generated from the prompts.
    """
    generated_images_dir = f"./data/generated_images/{model_id.replace('/', '_')}"
    os.makedirs(generated_images_dir, exist_ok=True)

    card_data = model_info(model_id).card_data.to_dict()
    tags = card_data['tags']

    if 'diffusers' not in tags:
        print('Model not supported.')
        return
    
    pipeline = None
    if 'base_model' in card_data.keys():
        base_model_id = card_data['base_model']
        
        print(f"Loading model {base_model_id} with LoRA weights from {model_id}...")

        pipeline = DiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16)
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)

        if base_model_id != model_id:
            pipeline.load_lora_weights(model_id)
        
        pipeline.to("cuda" if torch.cuda.is_available() else "cpu")

    else:
        pipeline = StableDiffusionPipeline.from_pretrained(model_id)

    pipeline.to("cuda" if torch.cuda.is_available() else "cpu")
    
    for cluster_id, items in clusters.items():
        cluster_dir = os.path.join(generated_images_dir, f"cluster_{cluster_id}")
        os.makedirs(cluster_dir, exist_ok=True)

        # Generate num_generated_images_per_cluster images for each cluster
        for i in range(image_generation_settings['num_generated_images_per_cluster']):
            # Cycle through items by using the modulus operator
            item = items[i % len(items)]
            
            # Construct the file path with a sequential index
            image_save_path = os.path.join(cluster_dir, f"generated_image_{i}.png")
            
            # Generate image for the selected prompt
            prompt = item['caption']
            print(f"\nGenerating image for prompt: '{prompt}'")
            image = pipeline(prompt, num_inference_steps=image_generation_settings['num_inference_steps']).images[0]
            image.save(image_save_path)
            print(f"Image saved to {image_save_path}")


