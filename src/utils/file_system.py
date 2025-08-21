import os
import sys
import yaml
import importlib.util
from pathlib import Path
from typing import Union, Optional, Dict, Any, List
from loguru import logger
from src.utils.user_interaction import *
from .putils import resolve, flatten_dict
import subprocess
import json
import os


# ---------- Path Utilities ----------
def get_project_root() -> Path:
    path = Path(__file__).resolve()
    while path != path.parent:
        if (path / 'README.md').exists() or (path / '.git').exists():
            return path
        path = path.parent
    raise FileNotFoundError("Project root not found.")


def abs_path(path: Union[str, Path]) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p

    root = os.getenv('ROOT')
    if root is None:
        root = str(get_project_root())
        
    return Path(root) / p

def check_file_exists(file_path: Union[str, Path], abs_path_convert = True) -> Path:
    if abs_path_convert:
        path = abs_path(file_path)
    else:
        path = Path(file_path)
    if not path.is_file():
        logger.error(f"File not found: {path}")
        raise FileNotFoundError(f"File not found: {path}")
    logger.success(f"File found: {path}")
    return path

def check_directory_exists(directory_path: Union[str, Path]) -> Path:
    path = abs_path(directory_path)
    if not path.is_dir():
        logger.error(f"Directory not found: {path}")
        raise NotADirectoryError(f"Directory not found: {path}")
    logger.success(f"Directory found: {path}")
    return path

# ---------- File and Module Handling ----------

def list_files_in_directory(pars: Dict[str, str]) -> List[Path]:
    dir_path = check_directory_exists(pars["dir"])
    return [f.relative_to(dir_path) for f in dir_path.rglob(pars['key']) if f.is_file()]

def yaml_load(yaml_path: Union[str, Path]) -> Dict[str, Any]:
    try:
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f) or {}

        pipeline_vars = flatten_dict(config.get("pipeline", {}))
        resolved_config = resolve(config, pipeline_vars)

        return resolved_config

    except Exception as e:
        logger.error(f"YAML load error: {e}")
        raise


def python_load(file_path: Union[str, Path]):
    path = abs_path(file_path)
    name = path.stem
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        logger.error(f"Python load error in '{path}': {e}")
        raise



def function_loader(conf: Dict[str, str], folder: Union[str, Path]):
    path = abs_path(folder) / (conf['function file'] + ".py")
    if not path.exists():
        logger.error(f"Function file missing: {path}")
        return None

    python_interpreter = conf.get("python_interpreter")

    if python_interpreter:
        def run_subprocess(input_dict):
            import os

            input_json = json.dumps(input_dict)
            env = os.environ.copy()
            env["PIPELINE_STEP_INPUT"] = input_json

            # Add project root to PYTHONPATH
            project_root = str(get_project_root())
            existing_pythonpath = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = f"{project_root}:{existing_pythonpath}"

            result = subprocess.run(
                [python_interpreter, str(path)],
                env=env,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                logger.error(f"Step {conf['function']} failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
            else:
                logger.info(f"Step {conf['function']} finished successfully.")
                logger.debug(f"STDOUT:\n{result.stdout}")

            return result.returncode == 0


        return run_subprocess

    # If no python_interpreter specified, load and return function inline
    module = python_load(path)
    return getattr(module, conf['function'], None)


# ---------- User Interaction Wrappers ----------

def file_select(pars: Dict[str, str]) -> str:
    files = [f for f in list_files_in_directory(pars) if f.name.lower() != '.env']
    if not files:
        raise FileNotFoundError(f"No {pars['key']} files found.")
    display_available_files(files, pars['key'])
    selected = get_user_choice(files, pars['key'])
    if not confirm_file_selection(selected):
        raise Exception(f"{pars['key']} selection canceled.")
    return selected

# ---------- Settings and Configuration ----------

def recursive_update(orig: Dict[str, Any], update: Dict[str, Any]) -> None:
    for k, v in update.items():
        if k not in orig:
            logger.warning(f"Key '{k}' not found. Adding.")
            orig[k] = v
            logger.success(f"Added '{k}'")
        elif isinstance(v, dict) and isinstance(orig.get(k), dict):
            recursive_update(orig[k], v)
        else:
            orig[k] = v
            masked = '****' if 'password' in k.lower() else v
            logger.success(f"Updated '{k}' to {masked}")

def update_settings(settings_raw: Dict[str, Any]) -> Dict[str, Any]:
    input_path = abs_path(settings_raw['input_file_path'])
    if input_path.suffix != ".py":
        logger.error("Input must be a Python file.")
        sys.exit(1)

    settings = python_load(input_path).processed_settings

    for k, v in settings_raw.items():
        if k in ('input_file_path', 'tweaks'):
            continue
        if isinstance(v, dict):
            settings.setdefault(k, {})
            recursive_update(settings[k], v)
        else:
            settings[k] = v
            logger.success(f"Set '{k}' to {v}")
    return settings

# ---------- Misc Utilities ----------

def find_subfolder(parent: Union[str, Path], name: str) -> Optional[Path]:
    folder = Path(parent) / name
    return folder if folder.is_dir() else None

def skip_if_condition(
    value: Union[str, Path],
    pdb_id: str = "",
    reason: str = "",
    log_handle: Any = logger,
    check: str = "absence",
    check_folder_or_file: str = "folder"
) -> bool:
    path = Path(value)
    file_or_folder = check_folder_or_file.lower().strip()

    if file_or_folder == "file":
        exists = path.is_file()
    elif file_or_folder == "folder":
        exists = path.is_dir()
    else:
        raise ValueError(f"Invalid 'check_folder_or_file': {check_folder_or_file}. Must be 'file' or 'folder'.")

    should_skip = (
        (check == "absence" and not exists) or
        (check == "existence" and exists)
    )

    if should_skip:
        msg = f"[SKIP] {pdb_id}: {reason}"
        print(msg)
        log_handle.info(msg + "\n")
        return True

    return False
