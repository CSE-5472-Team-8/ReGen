from huggingface_hub import HfApi
import yaml

def get_models_info(api):
    models = list(api.list_models(task="text-to-image", cardData=True, gated=False))
    return models

def find_model_dataset_pairs(api, models):
    model_dataset_pairs = []
    for model in models:
        card_data = yaml.safe_load(str(model.card_data))
        dataset = card_data.get('dataset', 'None')
        if dataset != 'None':
            try:
                api.dataset_info(dataset)
                model_dataset_pairs.append([model.id, dataset])
            except:
                continue
            
    return model_dataset_pairs

def run():
    api = HfApi()

    text_to_image_models = get_models_info(api)
    model_dataset_pairs = find_model_dataset_pairs(api, text_to_image_models)
    
    return model_dataset_pairs
