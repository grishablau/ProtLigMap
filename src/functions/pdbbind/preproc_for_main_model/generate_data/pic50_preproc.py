import re
import csv
import numpy as np
from pathlib import Path
from rdkit import Chem
from typing import Dict

from src.utils.file_system import check_directory_exists
from loguru import logger

# === Utility: Extract RMSD value from molecule props ===
def get_rmsd(mol):
    props = mol.GetPropsAsDict()
    remark = props.get("REMARK", "")
    match = re.search(r'VINA RESULT:\s+([-+]?[0-9.]+)\s+([-+]?[0-9.]+)\s+([-+]?[0-9.]+)', remark)
    if match:
        try:
            rmsd_lb = float(match.group(2))
            rmsd_ub = float(match.group(3))
            return (rmsd_lb + rmsd_ub) / 2.0
        except Exception:
            return None
    return None

def generate_pic50_labels(config: Dict[str, str]) -> None:
    """
    Generates pIC50 labels for docked ligand poses by combining known pKd values
    with pose-specific weights and RMSD values.

    This function:
    - Loads experimental pKd values from a text file (`pkd_nat_file`).
    - Iterates over subdirectories in `base_dir`, each representing a PDB complex.
    - For each complex, it:
        - Loads the corresponding docked SDF file.
        - Parses each pose, extracting its RMSD from the Vina result comment.
        - Loads the corresponding `.npz` file containing the `W` value (pose weight).
        - Computes the predicted pIC50 using the formula: `pIC50 = pKd_nat - W * RMSD`.
        - Stores pose names and their computed pIC50 values.
    - Writes all pIC50 labels to a `.tsv` file (`output_txt_path`).
    - Saves the pIC50 values as a dense `.npz` dictionary (`output_npz_path`).
    - Logs all errors (e.g., missing files, bad molecules, etc.) to a log file in `log_dir`.

    Args:
        config (Dict[str, str]): A dictionary with required paths:
            - "base_dir": Directory containing PDB-named subfolders with SDF files.
            - "w_data_dir": Directory containing corresponding `.npz` weight files.
            - "pkd_nat_file": Path to the file containing pKd values for each PDB code.
            - "output_txt_path": Path to write the tab-separated output text file.
            - "output_npz_path": Path to write the dense `.npz` file with pose-to-pIC50 mappings.
            - "log_dir": Directory to save the error log file.

    Notes:
        - SDF files are expected to be named `{prefix}_docked.sdf`.
        - Weight files are expected as `{prefix}_pose{i}.npz` and must contain a key `"W"`.
        - RMSD values are extracted from the "REMARK" field of the SDF pose using Vina's format.
        - The computed pIC50 = pKd - (W * RMSD) reflects penalization of experimental pKd by predicted flexibility.
    """

    base_dir = check_directory_exists(config["base_dir"])
    w_data_dir = check_directory_exists(config["w_data_dir"])
    pkd_nat_file = Path(config["pkd_nat_file"])
    output_txt_path = Path(config["output_txt_path"])
    output_npz_path = Path(config["output_npz_path"])
    log_dir = Path(config["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "pic50_errors.log"

    # Load pkd_nat data
    pkd_nat_dict = {}
    with open(pkd_nat_file, 'r') as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) >= 4:
                pdb_code = parts[0].lower()
                try:
                    pkd_nat_dict[pdb_code] = float(parts[3])
                except ValueError:
                    continue

    results = []
    dense_dict = {}

    with open(log_file, "w") as logf:
        for pdb_dir in Path(base_dir).iterdir():
            if not pdb_dir.is_dir():
                continue
            prefix = pdb_dir.name.lower()
            sdf_file = pdb_dir / f"{prefix}_docked.sdf"
            if not sdf_file.exists():
                logf.write(f"[Missing SDF] {prefix}\n")
                continue

            if prefix not in pkd_nat_dict:
                logf.write(f"[Missing pkd_nat] {prefix}\n")
                continue

            try:
                pkd_nat = pkd_nat_dict[prefix]
                npz_dir = w_data_dir / prefix
                suppl = Chem.SDMolSupplier(str(sdf_file), removeHs=False, sanitize=False)

                for i, mol in enumerate(suppl):
                    if mol is None:
                        logf.write(f"[Invalid mol] {prefix} pose {i}\n")
                        continue
                    rmsd = get_rmsd(mol)
                    if rmsd is None:
                        logf.write(f"[Missing RMSD] {prefix} pose {i}\n")
                        continue

                    npz_path = npz_dir / f"{prefix}_pose{i}.npz"
                    if not npz_path.exists():
                        logf.write(f"[Missing .npz file] {npz_path}\n")
                        continue

                    try:
                        data = np.load(npz_path)
                        w_val = data["W"].item()
                    except Exception as e:
                        logf.write(f"[Failed to read .npz] {npz_path}: {str(e)}\n")
                        continue

                    pkd_label = pkd_nat - w_val * rmsd
                    pose_name = f"{prefix}_pose{i}"
                    results.append((pose_name, pkd_label))
                    dense_dict[pose_name] = pkd_label

            except Exception as e:
                logf.write(f"[General error] {prefix}: {str(e)}\n")

    # Save results: Text
    with open(output_txt_path, "w") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["ligand_pose", "pIC50"])
        writer.writerows(results)

    # Save results: Dense NPZ
    np.savez_compressed(output_npz_path, **dense_dict)

    logger.info(f"Saved pIC50 data to:\n - {output_txt_path}\n - {output_npz_path}")
    logger.info(f"Errors logged to: {log_file}")

