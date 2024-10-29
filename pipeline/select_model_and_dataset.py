from huggingface_hub import HfApi
import get_model_and_dataset_pairs
import os
import csv

def get_valid_input(prompt, validate_func):
    """Get valid input from the user based on a validation function."""
    while True:
        user_input = input(prompt).strip()
        if validate_func(user_input):
            return user_input
        print("Invalid input, please try again.")

def validate_model_id(api, model_id):
    """Validate the model id using the Hugging Face API."""
    try:
        api.model_info(model_id)
        return True
    except Exception:
        return False

def validate_dataset_id(api, dataset_id):
    """Validate the dataset id using the Hugging Face API."""
    try:
        api.dataset_info(dataset_id)
        return True
    except Exception:
        return False

def manually_select_model_and_dataset_id(api):
    model_id = get_valid_input("\nEnter the text-to-image model id: ", lambda x: validate_model_id(api, x))
    dataset_id = get_valid_input("\nEnter the dataset id: ", lambda x: validate_dataset_id(api, x))
    return model_id, dataset_id

def load_or_fetch_model_dataset_pairs(file_path):
    """Load model-dataset pairs from CSV or fetch if not available."""
    if os.path.exists(file_path):
        with open(file_path, mode='r') as file:
            return list(csv.reader(file))
    else:
        print("Model id and dataset id pairs haven't been fetched yet. Please wait...")
        model_dataset_pairs = get_model_and_dataset_pairs.run()
        with open(file_path, mode='w', newline='') as file:
            csv.writer(file).writerows(model_dataset_pairs)
        return model_dataset_pairs

def display_model_dataset_options(model_dataset_pairs):
    """Display model and dataset options to the user."""
    index_width = 5
    first_column_width = max(len(model_id) for model_id, _ in model_dataset_pairs)
    second_column_width = max(len(dataset_id) for _, dataset_id in model_dataset_pairs)

    # Print header row
    print(" ".ljust(index_width) + "Model".ljust(first_column_width) + "Dataset".ljust(second_column_width))
    print("=" * (index_width + first_column_width + second_column_width))

    for index, (model_id, dataset_id) in enumerate(model_dataset_pairs, start=1):
        print(f"{(str(index) + ')').ljust(index_width)} {model_id.ljust(first_column_width)} {dataset_id.ljust(second_column_width)}")

def select_model_and_dataset_from_list(model_dataset_pairs):
    """Select model and dataset from a list."""
    display_model_dataset_options(model_dataset_pairs)
    valid_range = range(1, len(model_dataset_pairs) + 1)
    choice = get_valid_input("\nEnter your choice: ", lambda x: x.isdigit() and int(x) in valid_range)
    return model_dataset_pairs[int(choice) - 1]

def run():
    hf_api = HfApi()

    print("\n1) Manually choose a text-to-image model and dataset.")
    print("2) Select text-to-image model and dataset from list.")

    user_choice = get_valid_input("\nEnter your choice: ", lambda x: x in ["1", "2"])

    if user_choice == "1":
        while True:
            model_id, dataset_id = manually_select_model_and_dataset_id(hf_api)
            confirm = input(f"\nYou selected the model id '{model_id}' and dataset id '{dataset_id}'. Is that correct? (Y/N): ").strip().upper()
            if confirm == "Y":
                return model_id, dataset_id
    else:
        model_dataset_pairs_file_path = './data/model_dataset_pairs.csv'
        model_dataset_pairs = load_or_fetch_model_dataset_pairs(model_dataset_pairs_file_path)
        return select_model_and_dataset_from_list(model_dataset_pairs)
