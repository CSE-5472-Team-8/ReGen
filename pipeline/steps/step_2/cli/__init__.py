from steps.cli_helpers import get_valid_input, confirm_choice
from huggingface_hub import dataset_info

def setup_dataset(dataset_id):
    """
    Set up the dataset by detecting and confirming the required features based on user input.

    Args:
        dataset_id (str): The unique identifier of the dataset.

    Returns:
        dict: A dictionary mapping feature types (e.g., "image", "caption") to confirmed feature names, 
        either detected from the dataset or specified by the user.
    """
    dataset_type = select_dataset_type()
    
    features = fetch_dataset_features(dataset_id)
    dataset_feature_types = {
        "1": ["image", "caption"],
        "2": ["image_url", "caption"]
    }

    feature_types_to_detect = {feature_type: None for feature_type in dataset_feature_types.get(dataset_type, [])}
    feature_names = None

    if features:
        detected_feature_names = detect_features(features, feature_types_to_detect)
        feature_names = confirm_feature_detection(detected_feature_names)
    else:
        selected_feature_names = manually_select_features(feature_types_to_detect)
        feature_names = confirm_feature_detection(selected_feature_names)

    return feature_names


def select_dataset_type():
    """
    Prompt the user to select a dataset type and confirm their choice.

    Returns:
        str: The confirmed dataset type choice as a string.
    """
    options = [
        "Images and captions",
        "Image URLs and captions"
    ]

    print(f"\nThe dataset contains:")
    for i, option in enumerate(options, start=1):
        print(f"{i}) {option}")
    
    choice = get_valid_input("\nEnter the dataset type: ", lambda x: x in ["1", "2"])
    confirmed_choice = confirm_choice(f"You selected '{options[int(choice)-1]}'. Is this correct?", choice)
    
    return confirmed_choice if confirmed_choice else select_dataset_type()


def fetch_dataset_features(dataset_id):
    """
    Retrieve dataset features using the dataset ID.

    Args:
        dataset_id (str): The unique identifier of the dataset.

    Returns:
        list or None: A list of features if available; otherwise, None.
    """
    try:
        return dataset_info(dataset_id).card_data['dataset_info']['features']
    except:
        print("\nFailed to find dataset features. They need to be entered manually.")
        return None


def detect_features(features, feature_types_to_detect):
    """
    Detect relevant feature names in a dataset based on the feature types to detect.

    Args:
        features (list): A list of feature dictionaries, each containing 'dtype' and 'name' keys.
        feature_types_to_detect (dict): A dictionary where keys are feature types and values are initially None.

    Returns:
        dict: Updated feature_types_to_detect with detected feature names populated where found.
    """
    common_feature_names = {
        "image": ["image"],
        "image_url": ["url", "urls", "image_url", "image_urls"],
        "caption": ["text", "caption"]
    }

    for feature_type in feature_types_to_detect:
        expected_dtype = "image" if feature_type == "image" else "string"
        possible_names = common_feature_names[feature_type]

        for feature in features:
            if feature["dtype"] == expected_dtype and feature["name"] in possible_names:
                feature_types_to_detect[feature_type] = feature["name"]

    return feature_types_to_detect


def confirm_feature_detection(detected_features):
    """
    Confirm the detection of required features based on the detected keys. If only some features are detected,
    prompts user to confirm those, then manually select remaining features.

    Args:
        detected_features (dict): A dictionary where keys are feature types and values are detected names or None.

    Returns:
        dict: The confirmed or manually selected features.
    """
    all_detected = all(value is not None for value in detected_features.values())
    any_detected = any(value is not None for value in detected_features.values())

    if all_detected:
        message = build_confirmation_message(detected_features)
        confirmed_choice = confirm_choice(message, detected_features)
        
        if confirmed_choice:
            return confirmed_choice
        else:
            reset_features = {key: None for key in detected_features}
            return manually_select_features(reset_features)
    
    elif any_detected:
        detected_features_subset = {k: v for k, v in detected_features.items() if v is not None}
        message = build_confirmation_message(detected_features_subset)
        confirmed_choice = confirm_choice(message, detected_features)
        
        if confirmed_choice:
            return manually_select_features(confirmed_choice)
        else:
            reset_features = {key: None for key in detected_features}
            return manually_select_features(reset_features)
    
    else:
        print("No features were detected. Proceeding to manual selection.")
        return manually_select_features(detected_features)


def build_confirmation_message(features):
    """
    Build a custom confirmation message based on the detected features.

    Args:
        features (dict): A dictionary of detected features.

    Returns:
        str: A confirmation message string for the detected features.
    """
    feature_descriptions = []
    if "image" in features:
        feature_descriptions.append(f"image feature as '{features['image']}'")
    if "image_url" in features:
        feature_descriptions.append(f"image URL feature as '{features['image_url']}'")
    if "caption" in features:
        feature_descriptions.append(f"caption feature as '{features['caption']}'")
    
    return f"Detected {' and '.join(feature_descriptions)}. Is this correct?"


def manually_select_features(features):
    """
    Prompt the user to input feature names for any keys in the dictionary with a value of None.

    Args:
        features (dict): A dictionary of feature types where values are feature names or None.

    Returns:
        dict: The updated dictionary with user input for all None values.
    """
    for feature_type, feature_name in features.items():
        if feature_name is None:
            user_input = input(f"Enter the name of the {feature_type} feature: ").strip()
            features[feature_type] = user_input

    return features
