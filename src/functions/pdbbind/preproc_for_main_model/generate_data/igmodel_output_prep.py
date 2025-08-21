import numpy as np
from pathlib import Path
from rdkit import Chem
from typing import List, Dict
from src.utils.file_system import check_directory_exists
from models.backbone.IGModel.scripts.model import *
from src.utils.backbone_model_runner import collect_all_inputs, load_model, load_model_outputs
from loguru import logger
import os
import json
from typing import Optional
import sys
import inspect

def save_model_outputs_by_complex(inputs: List[List[str]], model, output_dir: Path, log_path: Path):
    with open(log_path, "w") as logf:
        for input_list in inputs:
            prefix = input_list[0].lower()
            sdf_file = input_list[3]

            try:
                output = load_model_outputs(input_list, model=model)
                feats = output["feats"].squeeze().cpu().numpy()
                W = output["W"].squeeze().cpu().numpy()
                suppl = Chem.SDMolSupplier(str(sdf_file), removeHs=False, sanitize=False)

                pdb_dir = output_dir / prefix
                pdb_dir.mkdir(parents=True, exist_ok=True)

                for i, mol in enumerate(suppl):
                    if mol is None:
                        logf.write(f"[Invalid mol] {prefix} pose {i}\n")
                        continue
                    if i >= len(feats) or i >= len(W):
                        logf.write(f"[Index error] {prefix} pose {i} > feats[{len(feats)}], W[{len(W)}]\n")
                        continue

                    pose_name = f"{prefix}_pose{i}"
                    out_path = pdb_dir / f"{pose_name}.npz"
                    np.savez_compressed(out_path, feats=feats[i], W=W[i])

            except Exception as e:
                logf.write(f"[General error] {prefix}: {str(e)}\n")

    logger.info(f"Saved features and weights organized by PDB in: {output_dir}")
    logger.info(f"Errors logged to: {log_path}")

def run_igmodel_feature_extraction(config: Dict[str, str]) -> None:
    """
    Runs IGModel feature extraction over a dataset of docked SDF files,
    saves model outputs (features and attention weights), and logs any issues.

    This function:
    - Loads the IGModel backbone.
    - Collects all input examples from the dataset path.
    - Runs inference for each SDF pose in the dataset using the model.
    - Saves the resulting features and attention weights into compressed `.npz` files,
      organized by PDB complex name.
    - Logs any errors (invalid molecules, model issues, I/O problems) to a log file.

    Args:
        config (Dict[str, str]): A dictionary with required keys:
            - "dataset_path": Path to the root directory containing complex subfolders with SDF files.
            - "output_dir": Path to the directory where model outputs (.npz files) will be saved.
            - "log_dir": Path to a directory where the error log will be written.

    Notes:
        - Output .npz files contain two arrays per pose: `feats` and `W`.
        - Errors are written to `igmodel_save_errors.log` inside `log_dir`.
        - The IGModel is assumed to be loaded from its internal default path.
    """

    current_function_name = inspect.currentframe().f_code.co_name
    print(f"\nPython Interpreter for the function {current_function_name}: {sys.executable}")


    dataset_path = check_directory_exists(config["dataset_path"])
    output_dir = check_directory_exists(config["output_dir"])
    log_dir = check_directory_exists(config["log_dir"])
    model_path  = config["model_path"]
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "igmodel_save_errors.log"
    




    model = load_model(model_path)
    inputs = collect_all_inputs(str(dataset_path))
    save_model_outputs_by_complex(inputs, model, output_dir, log_path)


def entry_point() -> Optional[int]:
    input_json = os.environ.get("PIPELINE_STEP_INPUT")
    if input_json is None:
        print("ERROR: PIPELINE_STEP_INPUT environment variable not set")
        return 1

    try:
        config = json.loads(input_json)
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse PIPELINE_STEP_INPUT JSON: {e}")
        return 1

    try:
        run_igmodel_feature_extraction(config)
    except Exception as e:
        print(f"ERROR: Exception during feature extraction: {e}")
        return 1

    return 0


entry_point()
