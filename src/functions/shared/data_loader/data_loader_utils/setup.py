from pathlib import Path
import sys
from loguru import logger

def setup_root_path(levels_up=5):
    try:
        notebook_path = Path(__file__).resolve()
    except NameError:
        # __file__ not defined (e.g. inside exec), use working directory
        notebook_path = Path.cwd()
    root_path = notebook_path.parents[levels_up - 1]
    sys.path.insert(0, str(root_path))
    logger.info(f"Root path set to: {root_path}")
    return root_path

root_path = setup_root_path()


# Import the full namespace as a short alias for ease of use in all step scripts
from src.functions.pdbbind.preproc_for_main_model.preproc import data_functions as data_preproc

logger.info("Notebook environment ready!")

globals()['data_preproc'] = data_preproc
