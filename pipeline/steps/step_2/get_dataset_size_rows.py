import requests
import os
from dotenv import load_dotenv
from cli.helpers import get_valid_input, confirm_choice

def run(dataset_id):
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