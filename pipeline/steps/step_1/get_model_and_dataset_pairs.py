from huggingface_hub import HfApi
import yaml

def get_models_info(api):
    """
    Retrieve metadata for all text-to-image models available on Hugging Face.

    Args:
        api (HfApi): The Hugging Face API instance.

    Returns:
        list: A list of models containing metadata for each text-to-image model.
    """
    models = list(api.list_models(task="text-to-image", cardData=True, gated=False))
    return models

def find_model_dataset_pairs(api, models):
    """
    Find and return model-dataset pairs by examining model metadata.

    Args:
        api (HfApi): The Hugging Face API instance.
        models (list): A list of models containing metadata.

    Returns:
        list: A list of model-dataset pairs, where each pair is represented by [model_id, dataset_id].
    """
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
    """
    Main function to execute the process of retrieving text-to-image models and their associated datasets.

    Returns:
        list: A list of model-dataset pairs.
    """
    api = HfApi()

    text_to_image_models = get_models_info(api)
    model_dataset_pairs = find_model_dataset_pairs(api, text_to_image_models)
    return model_dataset_pairs