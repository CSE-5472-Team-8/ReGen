def get_valid_input(prompt, validate_func):
    """
    Prompt the user for input and validate the response.
    Keeps prompting until the user provides valid input as determined by the validation function.

    Args:
        prompt (str): The message to display to the user.
        validate_func (function): A function that returns True if the input is valid, False otherwise.

    Returns:
        str: The validated user input.
    """
    while True:
        choice = input(prompt).strip()
        if validate_func(choice):
            return choice
        print("Invalid input, please try again.")

def confirm_choice(message, data):
    """
    Confirm the user's selection by displaying a message and awaiting confirmation.

    Args:
        message (str): The message to display to the user.
        data (list): The data to return if the user confirms.

    Returns:
        list: The data if confirmed by the user, otherwise None.
    """
    while True:
        confirm = input(f"\n{message} (Y/N): ").strip().upper()
        if confirm == "Y":
            return data
        else:
            return None
