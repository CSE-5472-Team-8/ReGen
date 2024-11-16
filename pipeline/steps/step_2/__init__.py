import requests
import os
from dotenv import load_dotenv
from steps.cli_helpers import get_valid_input, confirm_choice

def get_dataset_size_rows(dataset_id):
    """
    Retrieve the number of rows in a dataset using the Hugging Face API.
    If the API call fails or returns incomplete data, prompt the user for manual input.
    
    Args:
        dataset_id (str): The identifier for the dataset on Hugging Face.
    
    Returns:
        int or None: The number of rows in the dataset, either fetched from the API 
                     or input manually by the user. Returns None if user input is not confirmed.
    """
    
    load_dotenv()

    headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_TOKEN')}"}
    API_URL = f"https://datasets-server.huggingface.co/size?dataset={dataset_id}"

    data = None
    response = requests.get(API_URL, headers=headers)
    
    if response.status_code == 200:
        data = response.json()

    if data and 'dataset' in data['size']:
        num_rows = data['size']['dataset']['num_rows']
        return num_rows
    else:
        # Prompt user to input number of rows if API call fails
        num_rows = get_valid_input(
            "Enter the number of rows in the dataset: ", 
            lambda x: x.isdigit()
        )
        
        # Confirm the user's input
        confirmed_rows = confirm_choice(
            f"You entered {num_rows} rows. Is that correct?", num_rows
        )
        
        return int(confirmed_rows) if confirmed_rows else None