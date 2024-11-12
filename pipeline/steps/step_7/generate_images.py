from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

def generate_images(model_id, prompts):
    """
    Generate images from a text-to-image model on Hugging Face using the provided prompts.
    
    Args:
    - model_id (str): The model ID on Hugging Face (e.g., 'CompVis/stable-diffusion-v1-4').
    - prompts (list): A list of prompts (strings) for image generation.
    
    Returns:
    - images (list): A list of PIL Image objects generated from the prompts.
    """
    pipeline = StableDiffusionPipeline.from_pretrained(model_id)
    pipeline = pipeline.to("cuda" if torch.cuda.is_available() else "cpu")
    
    generated_images = []
    
    for prompt in prompts:
        image = pipeline(prompt).images[0]
        generated_images.append(image)
        
    return generated_images

model_id = "CompVis/stable-diffusion-v1-4"
prompts = ["A sunset over a mountain range", "A futuristic cityscape with flying cars"]
generated_images = generate_images(model_id, prompts)

for idx, img in enumerate(generated_images):
    img.save(f"generated_image_{idx}.png")
    img.show()
