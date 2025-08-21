import sys
import logging
from pathlib import Path
from loguru import logger

# --- Determine key paths
pipeline_run_path = Path(__file__).resolve()
parent_folder = pipeline_run_path.parents[0]
project_root = parent_folder.parents[3]
functions_dir = parent_folder / "data_loader_utils"
config_path = project_root / "config/settings/data_loader/pipeline.py"


exec_scope = {
    "__builtins__": __builtins__,
    "sys": sys,
    "Path": Path,
    "logger": logger,
    "logging": logging,
    "__file__": str(pipeline_run_path),  
}


# --- Function factory: turns .py files into callable steps
def make_step_runner(script_path):
    def run_step(*args, **kwargs):
        local_scope = {"args": args, "kwargs": kwargs}
        exec(script_path.read_text(), exec_scope, local_scope)
        return local_scope.get("result", None)
    return run_step

# --- Register all functions dynamically from pipeline_functions
for script_path in functions_dir.glob("*.py"):
    step_name = script_path.stem
    if step_name.startswith("__"):
        continue
    exec_scope[step_name] = make_step_runner(script_path)

# --- Run the pipeline config script (which calls e.g., `setup()`, `load_config()`)
exec(config_path.read_text(), exec_scope)

# --- Make the variables visible to the notebook
globals().update(exec_scope)
