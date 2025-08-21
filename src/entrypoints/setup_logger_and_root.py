import sys
import os
from pathlib import Path
from loguru import logger
from datetime import datetime

def setup_logger_and_root(script_path: Path = None) -> Path:
    """
    Set up the logger to save logs in a unique timestamped subfolder within 'logs',
    relative to the script location. Also sets the 'ROOT' environment variable.
    Returns the root directory path.
    """

    # Default to the current file if no script_path is provided
    if script_path is None:
        script_path = Path(__file__).resolve()
    else:
        script_path = Path(script_path).resolve()

    # Define root directory
    root_dir = script_path.parent
    os.environ["ROOT"] = str(root_dir)

    # Create logs root folder
    logs_root_dir = root_dir / "logs"
    logs_root_dir.mkdir(parents=True, exist_ok=True)

    # Create a subfolder based on current datetime
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_log_dir = logs_root_dir / f"run_{timestamp}"
    run_log_dir.mkdir(parents=True, exist_ok=True)

    # Path to the actual log file
    log_path = run_log_dir / "pipeline.log"

    # Configure logger
    logger.add(
        str(log_path),
        level="DEBUG",
        rotation="10 MB",
        backtrace=True,
        diagnose=True
    )

    logger.debug(f"Logger initialized. Python executable: {sys.executable}")
    logger.debug(f"Logs are being written to: {log_path}")
    
    return root_dir
