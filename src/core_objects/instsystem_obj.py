from src.utils.file_system import *
from loguru import logger
from src.utils.putils import to_dict


class InstConf:
    """A class to manage the configurations of the instructions."""

    def __init__(self, env):
        # env
        # Get the instructions/pipeline folder path from the env basic settings. You will you use it to get the default
        # instructions for each step in the instructions
        self.instructions_folder = env.instructions_folder
        # Set the root path of the instructions object. Get it from the current env root
        self.root_path = Path(env.root)
        # Set the keyword to search with for files of instructions/pipelines
        self.instructions_file_pattern = env.instructions_file_pattern

        # other
        # Set the full instructions/pipeline path
        self.full_instructions_path = None

    def get_instructions_full_path(self):
        """Returns the full path of the selected pipeline file."""
        # Construct the directory full path
        directory_full_path = self.root_path
        # If instruction folder exists in the env settings, us it
        if self.instructions_folder:
            directory_full_path = directory_full_path / self.instructions_folder
        # If INSTRUCTIONS_AUTO is set to False, then use the pipeline selector to select the right instructions
        if os.getenv('INSTRUCTIONS_AUTO', '').upper() == "FALSE":
            file_selection_params = {"dir": directory_full_path, "key": self.instructions_file_pattern}
            # Use the pipeline_select method to get the selected pipeline
            selected_pipeline = file_select(file_selection_params)
        # If INSTRUCTIONS_AUTO is set to True, then get the path from INSTRUCTIONS_PATH
        elif os.getenv('INSTRUCTIONS_AUTO', '').upper() == "TRUE":
            if os.getenv('INSTRUCTIONS_PATH'):
                selected_pipeline = os.getenv('INSTRUCTIONS_PATH')
            else:
                logger.error("ERROR: INSTRUCTIONS_PATH environment variable must be set in .env!")
                sys.exit(1)  # Exit the program with a non-zero status indicating an error
        else:
            # If neither TRUE nor FALSE is found, raise an error and exit
            logger.error("ERROR: INSTRUCTIONS_AUTO environment variable must be set to 'TRUE' or 'FALSE'.")
            sys.exit(1)  # Exit the program with a non-zero status indicating an error

        # Get the full pipeline path
        self.full_instructions_path = directory_full_path / selected_pipeline


class StepObj:
    """A class to manage a single step as part of the whole instructions."""

    def __init__(self, conf):
        self.info = conf  # Assuming 'info' is part of conf
        self.tweaks = []

    def get_nested_value(self, key_tree):
        """Helper function to traverse self.info based on key_tree."""
        if not key_tree:
            return self.info  # If key_tree is empty, return self.info itself
        # Traverse through the dictionary based on the key_tree
        result = self.info
        for key in key_tree:
            if isinstance(result, dict) and key in result:
                result = result[key]  # Move to the next level of the dictionary
            else:
                # If key doesn't exist or it's not a dictionary, return None or handle error
                return None
        return result

    def update_info_mask_password(self, tweaks, key_tree, parent_file_path, child_file_path):
        """Helper function to mask password fields for logging and updating new settings in general"""
        """This function goes recursively through the new settings, saves the tree hierarchy that it goes in through
         in both the new and the original settings, and updates if any needed to be updated"""
        for key, value in tweaks.items():
            original_value = self.get_nested_value(key_tree)
            key_tree.append(key)
            
            if key not in original_value:
                logger.warning(f"'{key}' settings from updates settings or the parent pipeline/step file which is : {parent_file_path} doesn't exist at all or doesn't exist in"
                               f" the right format in the original default settings step file which is {child_file_path}")
                logger.warning(f"...adding '{key}' settings from updates settings file into the final settings, extra to"
                               f" the default settings")
                original_value[key] = value
                self.tweaks.append(key_tree.copy())
                key_tree.pop()
            else:
                if isinstance(value, dict) and isinstance(original_value[key], dict):
                    self.update_info_mask_password(value, key_tree, parent_file_path, child_file_path)
                    key_tree.pop()
                elif isinstance(value, (str, bool)) or value is None:
                    if type(original_value[key]) == type(value):
                        original_value[key] = value
                        masked_value = value
                        if 'password' in key.lower():
                            masked_value = '****'
                        logger.success(
                            f"function {self.info.get('function', 'N/A')}: Successfully updated '{key}' to {masked_value}"
                        )
                        self.tweaks.append(key_tree.copy())
                        key_tree.pop()
                    else:
                        logger.warning(
                            f"function '{self.info.get('function', 'N/A')}': You tried to update '{key}' to '{value}',"
                            f" of the type '{type(value).__name__}' but \
'{key}' value should be of the type '{type(original_value[key]).__name__}'"
                        )


class InstObj:
    """A class to manage the instructions themselves."""

    def __init__(self):
        # The basic instruction/pipeline (might be overwritten, but this is the basis)
        self.instructions_raw = None
        # The steps of the final instruction/pipeline
        self.steps = []

    def preproc(self, raw_full_path):
        """Preprocess the pipeline by loading the raw instructions from the given file."""
        # Get the instructions directly from the instructions file
        self.instructions_raw = yaml_load(raw_full_path)  # Assuming yaml_load loads the raw YAML data
        # Reset the steps list before populating it
        self.steps = []
        # Get the steps to run including both their names and possible tweaks
        steps_to_run = self.instructions_raw['pipeline']['steps']
        # If there are steps to run, otherwise give an error
        if steps_to_run:
            # Ensure steps_to_run is dict
            steps_to_run = to_dict(steps_to_run)
            # Go over the steps and prepare their instructions
            for step_name, step_tweaks in steps_to_run.items():
                # Step 1 - Get the general settings folder path of all the steps
                steps_folder = self.instructions_raw['pipeline']['pipeline steps folder']
                # Step 2 - Load the default settings for the given step using its name and general steps folder
                yaml_file_path_abs = abs_path(os.path.join(steps_folder, f'{step_name}.yaml'))
                # Step 3 - Load the step default instructions
                if os.path.exists(yaml_file_path_abs):
                    step_default_instructions = yaml_load(str(yaml_file_path_abs))
                    # Step 4 - Create object according to the default settings
                    step_obj = StepObj(step_default_instructions)
                    # Step 5 - If there are any new settings for the current step, update the settings on top of
                    # the default ones
                    if step_tweaks:
                        step_tweaks = to_dict(step_tweaks)
                        step_obj.update_info_mask_password(step_tweaks, [], raw_full_path, yaml_file_path_abs)
                    # Step 6 - add the tweaks to 'input' of the step
                    if step_obj.info['input'] is None:
                        step_obj.info['input'] = {}  # Assign an empty dictionary if the input is None

                    # Ensure input is a dictionary
                    if not isinstance(step_obj.info['input'], dict):
                        try:
                            step_obj.info['input'] = dict(step_obj.info['input'])
                        except Exception as e:
                            logger.error(f"Error converting input to dict: {e}")

                    # Add the 'tweaks' to the input dictionary
                    step_obj.info['input']['tweaks'] = step_obj.tweaks

                    # Step 7 - Append the step to the list of steps
                    self.steps.append(step_obj)  # Add the StepObj to the steps list
                else:
                    logger.error(f"{yaml_file_path_abs} does not exist!")
                    sys.exit(1)
        else:
            logger.error(f"Error: no steps listed in instruction file!")
            sys.exit(1)


class InstSystem:
    """A class to manage the entire instructions system, including environment and instructions/pipeline."""

    def __init__(self, env):
        # Initialize the environment settings of the instructions/pipeline
        self.inst_env = InstConf(env)
        # Initialize the object of the instructions/pipeline = literally the instructions
        self.inst_obj = InstObj()

    def pre_run_proc(self):
        # Get the instructions/pipeline full path
        self.inst_env.get_instructions_full_path()
        # Unite all the instructions into one, ready to run using the instructions/pipeline file
        self.inst_obj.preproc(self.inst_env.full_instructions_path)
