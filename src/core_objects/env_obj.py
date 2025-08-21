import os
from dotenv import load_dotenv
from src.entrypoints.env_config import get_root
from src.utils.file_system import get_project_root, file_select, abs_path
from loguru import logger

# One underscore for internal use only
def _load_environment_variables():
    """Load environment variables from the .env file."""
    # Load .env settings
    load_dotenv()
    # Set the root folder as the absolute root folder for the current run
    if not os.environ["ROOT"]:
        os.environ["ROOT"] = str(get_project_root())
    # If ENV_AUTO is NOT set to true it means that we are NOT going to run the environment in default settings
    if not os.getenv('ENV_AUTO').upper() == 'TRUE':
        del os.environ['ENV_MODE']
        file_selection_params = {"dir": get_root(), "key": '*.env'}
        env_type = abs_path(file_select(file_selection_params))
        print(env_type)
        load_dotenv(dotenv_path=env_type)
        logger.info(f"Loaded environment variables from: {env_type.parts[-1]}")


class Environment:
    """
    The most basic default env object.

    Using @property decorator method that behaves like an attribute,
    which means you can access it without calling it like a function.
    """
    def __init__(self):
        # Initialise the default variables of the env
        _load_environment_variables()
    
    @property
    def instructions_folder(self) -> str:
        """Get the pipeline folder path from environment variables."""
        return os.getenv("INSTRUCTIONS_FOLDER")

    @property
    def instructions_file_pattern(self) -> str:
        """Get the pipeline file pattern from environment variables."""
        return os.getenv("INSTRUCTIONS_FILE_PATTERN")

    @property
    def root(self) -> str:
        """Get the root path from environment variables."""
        return os.getenv("ROOT")
