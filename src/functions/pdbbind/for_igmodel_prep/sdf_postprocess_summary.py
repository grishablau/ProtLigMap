import csv
import json
from pathlib import Path
from typing import Dict, List
from rdkit import Chem
from rdkit.Chem import rdmolfiles, rdmolops

from src.utils.file_system import check_directory_exists, check_file_exists
from src.utils.subprocess_utils import safe_run
from loguru import logger

def file_exists_and_nonempty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0

def sanitize_molecule(mol):
    try:
        rdmolops.SanitizeMol(mol)
        return mol
    except Exception:
        return None

def filter_and_rewrite_sdf(input_sdf_path: Path, output_sdf_path: Path) -> int:
    suppl = Chem.SDMolSupplier(str(input_sdf_path), removeHs=False, sanitize=False)
    valid_mols = []

    for i, mol in enumerate(suppl):
        if mol is None:
            continue
        sanitized = sanitize_molecule(mol)
        if sanitized is not None:
            valid_mols.append(sanitized)
        else:
            logger.warning(f"Skipping molecule {i} due to sanitization error.")

    if not valid_mols:
        return 0

    writer = rdmolfiles.SDWriter(str(output_sdf_path))
    for mol in valid_mols:
        writer.write(mol)
    writer.close()

    return len(valid_mols)

def postprocess_docking_outputs(config: Dict[str, str]) -> None:
    """
    Postprocess docking outputs by:
    1. Converting .pdbqt docked files to raw .sdf using Open Babel.
    2. Filtering and sanitizing molecules using RDKit.
    3. Writing summary of valid/invalid outputs.

    Args:
        config (dict): Configuration dictionary containing:
            - "dataset_path": str, root folder with complex subdirectories
            - "log_folder": str, folder to write summary files
    """
    dataset_root = check_directory_exists(config["dataset_path"])
    log_folder = check_directory_exists(config["log_folder"])
    excluded_records: List[Dict] = []
    summary_records: List[Dict] = []

    for complex_dir in dataset_root.iterdir():
        if not complex_dir.is_dir():
            continue

        complex_id = complex_dir.name
        logger.info(f"Processing {complex_id}")

        decoy_file = complex_dir / f"{complex_id}_docked.pdbqt"
        sdf_output_raw = complex_dir / f"{complex_id}_docked_raw.sdf"
        sdf_output_clean = complex_dir / f"{complex_id}_docked.sdf"
        warning_msgs = []

        if not file_exists_and_nonempty(decoy_file):
            logger.warning(f"Docked file missing or empty: {decoy_file}")
            excluded_records.append({
                "complex_id": complex_id,
                "reason": "Docked PDBQT missing or empty"
            })
            continue

        if not file_exists_and_nonempty(sdf_output_raw):
            cmd = [
                "obabel", str(decoy_file), "-O", str(sdf_output_raw),
                "--multiple", "--keep3d", "-h"
            ]
            if not safe_run(cmd, desc=f"Converting {decoy_file.name} to raw SDF"):
                warning_msgs.append("OpenBabel conversion failed")
                continue
        else:
            logger.info(f"Raw SDF already exists: {sdf_output_raw}")

        if not file_exists_and_nonempty(sdf_output_clean):
            num_valid_mols = filter_and_rewrite_sdf(sdf_output_raw, sdf_output_clean)
            if num_valid_mols == 0:
                logger.error(f"No valid molecules after sanitization in {sdf_output_raw}")
                excluded_records.append({
                    "complex_id": complex_id,
                    "reason": "All molecules failed sanitization"
                })
                continue
        else:
            logger.info(f"Clean SDF already exists: {sdf_output_clean}")

        summary_records.append({
            "complex_id": complex_id,
            "sdf_valid_structure": True,
            "sdf_valid_rdkit": True,
            "warnings": "; ".join(warning_msgs) if warning_msgs else ""
        })

    if summary_records:
        csv_path = log_folder / "sdf_summary.csv"
        json_path = log_folder / "sdf_summary.json"
        with open(csv_path, "w", newline="") as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=summary_records[0].keys())
            writer.writeheader()
            writer.writerows(summary_records)
        with open(json_path, "w") as f_json:
            json.dump(summary_records, f_json, indent=2)
        logger.info(f"Summary saved: {csv_path}, {json_path}")
    else:
        logger.warning("No valid records to summarize.")

    if excluded_records:
        excluded_path = log_folder / "excluded_complexes_due_to_invalid_molecules.log"
        with open(excluded_path, "w") as excl_f:
            for record in excluded_records:
                excl_f.write(f"{record['complex_id']}: {record['reason']}\n")
        logger.info(f"Excluded complexes saved: {excluded_path}")
