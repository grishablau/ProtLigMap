from pathlib import Path
from typing import Dict
import shutil
from loguru import logger
from src.utils.file_system import abs_path

def copy_sdf_files(config: Dict[str, str]) -> None:
    """
    Copies ligand SDF files from source folders to matching destination folders
    based on PDB folder names.

    Args:
        config (Dict[str, str]): A dictionary with the following keys:
            - "sdf_source": Path to the root directory containing folders named by PDB IDs,
                            each with a '{pdb_id}_ligand.sdf' file inside.
            - "sdf_destination": Path to the root directory where folders named by PDB IDs
                                 already exist. The ligand SDFs will be copied into these folders
                                 under the name f"{pdb_id}_ligand.sdf"

    Raises:
        ValueError: If any of the required keys are missing in the config.
    """
    sdf_source = config.get("sdf_source")
    sdf_destination = config.get("sdf_destination")

    if not sdf_source or not sdf_destination:
        raise ValueError("Both 'sdf_source' and 'sdf_destination' must be provided in config.")

    sdf_source = abs_path(Path(sdf_source))
    sdf_destination = abs_path(Path(sdf_destination))

    logger.info(f"Starting SDF copy from {sdf_source} to {sdf_destination}")

    for dest_subfolder in sdf_destination.iterdir():
        if not dest_subfolder.is_dir():
            continue

        pdb_id = dest_subfolder.name.lower()
        sdf_filename = f"{pdb_id}_ligand.sdf"
        source_sdf_path = sdf_source / pdb_id / sdf_filename
        dest_sdf_path = dest_subfolder / f"{pdb_id}_ligand.sdf"

        if source_sdf_path.exists():
            shutil.copy(source_sdf_path, dest_sdf_path)
            logger.info(f"Copied {source_sdf_path} to {dest_sdf_path}")
        else:
            logger.warning(f"[SKIP] {sdf_filename} not found in {sdf_source / pdb_id}")

    logger.info("Finished copying SDF files.")
