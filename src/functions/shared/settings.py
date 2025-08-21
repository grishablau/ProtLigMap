import yaml
from pathlib import Path
from src.utils.file_system import abs_path
from loguru import logger

# Function to load and process settings
def load_and_process_settings_yaml(input_dict):
    input_file_path, save = (input_dict[k] for k in ("input_file_path", "save"))
    tweaks = input_dict.get("tweaks")  # This will be None if "tweaks" doesn't exist

    """Converts a YAML file to a Python object and saves it as a .py file."""
    input_file_path_abs = abs_path(input_file_path)
    # Load the YAML content
    with open(input_file_path_abs, 'r') as file:
        yaml_data = yaml.safe_load(file)
    # Convert the YAML data into a Python object (usually a dictionary)
    processed_settings = yaml_data  # You can further process this data if needed
    save_path = save
    auto_save_path = True
    if tweaks:
        if ['input', 'save'] in tweaks:
            auto_save_path = False
            save_path = abs_path(save)

    if auto_save_path:
        # Extract the base filename from input_file (without extension)
        # Remove extension from filename2 and combine with file_name_from_path1
        combined_filename = f"{Path(save).stem}.py"
        # Save the processed settings as a Python file
        save_path = Path(input_file_path_abs).resolve().parent.parent.parent / 'processed' / combined_filename

    with open(save_path, 'w') as python_file:
        python_file.write(f"processed_settings = {processed_settings}")
    logger.success(f"Settings have been processed and saved to {save_path}")
    # Return the processed settings as a Python object
    return processed_settings
