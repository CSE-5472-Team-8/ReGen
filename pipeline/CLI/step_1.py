from huggingface_hub import HfApi
from step_1 import get_model_and_dataset_pairs
import os
import csv
from CLI.helpers import get_valid_input, confirm_choice

def validate_model_id(api, model_id):
    """
    Validate the model ID using the Hugging Face API.

    Args:
        api (HfApi): The Hugging Face API instance.
        model_id (str): The model ID to validate.

    Returns:
        bool: True if the model ID is valid, False otherwise.
    """
    try:
        api.model_info(model_id)
        return True
    except Exception:
        return False

def validate_dataset_id(api, dataset_id):
    """
    Validate the dataset ID using the Hugging Face API.

    Args:
        api (HfApi): The Hugging Face API instance.
        dataset_id (str): The dataset ID to validate.

    Returns:
        bool: True if the dataset ID is valid, False otherwise.
    """
    try:
        api.dataset_info(dataset_id)
        return True
    except Exception:
        return False

def confirm_model_and_dataset_choice(model_id, dataset_id):
    """
    Confirm the user's selection of the model and dataset IDs.

    Args:
        model_id (str): The selected model ID.
        dataset_id (str): The selected dataset ID.

    Returns:
        tuple: Returns the model ID and dataset ID if confirmed, otherwise None.
    """
    message = f"You selected the model ID '{model_id}' and dataset ID '{dataset_id}'. Is that correct?"
    return confirm_choice(message, [model_id, dataset_id])

def manually_select_model_and_dataset_id(api):
    """
    Prompt the user to manually enter a model ID and dataset ID.
    Ensures that both IDs are valid by using the Hugging Face API and confirms the selection.

    Args:
        api (HfApi): The Hugging Face API instance.

    Returns:
        tuple: A tuple containing the valid model ID and dataset ID if confirmed.
    """
    model_id = get_valid_input("\nEnter the text-to-image model ID: ", lambda x: validate_model_id(api, x))
    dataset_id = get_valid_input("\nEnter the dataset ID: ", lambda x: validate_dataset_id(api, x))
    return confirm_model_and_dataset_choice(model_id, dataset_id)

def load_or_fetch_model_dataset_pairs(file_path):
    """
    Load model-dataset pairs from a CSV file or fetch them if the file does not exist.
    If fetching is necessary, the fetched pairs are saved to the CSV file.

    Args:
        file_path (str): The path to the CSV file containing model-dataset pairs.

    Returns:
        list: A list of model-dataset pairs, where each pair is a tuple of (model_id, dataset_id).
    """
    if os.path.exists(file_path):
        with open(file_path, mode='r') as file:
            return list(csv.reader(file))
    else:
        print("Model ID and dataset ID pairs haven't been fetched yet. Please wait...")
        model_dataset_pairs = get_model_and_dataset_pairs.run()
        with open(file_path, mode='w', newline='') as file:
            csv.writer(file).writerows(model_dataset_pairs)
        return model_dataset_pairs

def display_model_dataset_options(model_dataset_pairs):
    """
    Display a list of available model-dataset pairs to the user in a formatted table.

    Args:
        model_dataset_pairs (list): A list of model-dataset pairs to display.
    """
    index_width = 5
    first_column_width = max(len(model_id) for model_id, _ in model_dataset_pairs)
    second_column_width = max(len(dataset_id) for _, dataset_id in model_dataset_pairs)

    print(" ".ljust(index_width) + "Model".ljust(first_column_width) + "Dataset".ljust(second_column_width))
    print("=" * (index_width + first_column_width + second_column_width))

    for index, (model_id, dataset_id) in enumerate(model_dataset_pairs, start=1):
        print(f"{(str(index) + ')').ljust(index_width)} {model_id.ljust(first_column_width)} {dataset_id.ljust(second_column_width)}")

def select_model_and_dataset_from_list(model_dataset_pairs):
    """
    Allow the user to select a model-dataset pair from a list of available options.
    Confirms the selection with the user.

    Args:
        model_dataset_pairs (list): A list of model-dataset pairs to choose from.

    Returns:
        tuple: The selected model-dataset pair if confirmed, otherwise None.
    """
    while True:
        display_model_dataset_options(model_dataset_pairs)
        valid_range = range(1, len(model_dataset_pairs) + 1)
        choice = get_valid_input("\nEnter your choice: ", lambda x: x.isdigit() and int(x) in valid_range)
        model_id, dataset_id = model_dataset_pairs[int(choice) - 1]
        confirmed = confirm_model_and_dataset_choice(model_id, dataset_id)
        
        if confirmed:
            return confirmed

def select_model_and_dataset():
    """
    Guide the user to select a text-to-image model and dataset, either manually or from a pre-defined list.

    Returns:
        tuple: The selected model ID and dataset ID.
    """
    api = HfApi()

    print("\n1) Manually choose a text-to-image model and dataset.")
    print("2) Select text-to-image model and dataset from list.")

    choice = get_valid_input("\nEnter your choice: ", lambda x: x in ["1", "2"])

    if choice == "1":
        while True:
            model_id, dataset_id = manually_select_model_and_dataset_id(api)
            if model_id and dataset_id:
                return model_id, dataset_id
    else:
        model_dataset_pairs_file_path = './data/model_dataset_pairs.csv'
        model_dataset_pairs = load_or_fetch_model_dataset_pairs(model_dataset_pairs_file_path)
        model_id, dataset_id = select_model_and_dataset_from_list(model_dataset_pairs)
        if model_id and dataset_id:
            return model_id, dataset_id
