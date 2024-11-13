import json

with open('./config/settings.json', 'r') as file:
    settings = json.load(file)

clip_embeddings_settings = settings.get("clip_embeddings", {})
find_clusters_settings = settings.get("find_clusters", {})
image_generation_settings = settings.get("image_generation", {})
image_comparison_settings = settings.get("image_comparison", {})