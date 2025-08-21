import os
from pathlib import Path
from dotenv import load_dotenv
from src.utils.file_system import get_project_root


def load_environment_variables():
    """Load environment variables from the .env file."""
    load_dotenv()
    os.environ["ROOT_PATH"] = str(get_project_root())


def get_pipeline_folder() -> str:
    """Get the pipeline_old folder path from environment variables."""
    return os.getenv("PIPELINE_FOLDER", "config/pipelines/")


def get_pipeline_file_pattern() -> str:
    """Get the pipeline_old folder path from environment variables."""
    return os.getenv("PIPELINE_FILE_PATTERN")


def get_root() -> Path:
    """Get the pipeline_old folder path from environment variables."""
    return Path(os.getenv("ROOT"))
