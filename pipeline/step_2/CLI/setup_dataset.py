from step_2 import get_dataset_size

def run(dataset_id):

    dataset_size = get_dataset_size.run(dataset_id)
    
    execution_mode = None

    # "The dataset '{dataset_id}' contains:
    # 1) Images and captions (fastest download speed, but images can't be easily fetched either -- need to do something like save the index)
    # 2) Images, image urls, and captions (best overall - can store image urls in metadata to easily fetch the image later, fastest download speed)
    # 3) Image urls and captions (slowest download speed, but can easily fetch the image later)""

    # Then depending on their dataset type selection, prompt them to say which field contains the caption, image?, image_url?

    # Download the data if its <= 2 GB or some other manageable amount of memory. (Maybe let the user decide with a config file or something?)
    # Otherwise stream it

    # Programatically choose the execution mode based on the dataset type and whether or not it can be streamed.

    return execution_mode