import requests
import os
from dotenv import load_dotenv
from huggingface_hub import dataset_info

def run(dataset_id):
    load_dotenv()
    
    headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_TOKEN')}"}
    API_URL = f"https://datasets-server.huggingface.co/size?dataset={dataset_id}"

    data = None
    response = requests.get(API_URL, headers=headers)
    if response.status_code == 200:
        data = response.json()

    if data and 'splits' in data['size']:
        splits = data['size']['splits']
        num_bytes = sum([ split['num_bytes_memory'] for split in splits if 'num_bytes_memory' in split ])

        # Convert to MB and GB
        num_megabytes = num_bytes / (1024 ** 2)  # Divide by 1,048,576
        num_gigabytes = num_bytes / (1024 ** 3)  # Divide by 1,073,741,824
        num_terabytes = num_bytes / (1024 ** 4)  # TB

        return { "bytes": num_bytes, "megabytes": num_megabytes, "gigabytes": num_gigabytes, "terabytes": num_terabytes }
    
    else:
        return None