import os
import shutil
from loguru import logger
from rdkit import Chem
from rdkit import RDLogger
from typing import Dict


# Redirect RDKit warnings to logger
class RDKitLogHandler:
    def write(self, message):
        msg = message.strip()
        if msg:
            logger.warning(f"RDKit: {msg}")
    def flush(self):
        pass

import sys
sys.stderr = RDKitLogHandler()
RDLogger.logger().setLevel(RDLogger.WARNING)


def load_ligand_mol(folder_path: str):
    for fname in os.listdir(folder_path):
        if fname.endswith('.mol2') or fname.endswith('.sdf'):
            file_path = os.path.join(folder_path, fname)
            try:
                if fname.endswith('.mol2'):
                    mol = Chem.MolFromMol2File(file_path, sanitize=False)
                else:
                    mol = Chem.MolFromMolFile(file_path, sanitize=False)

                if mol is None:
                    logger.warning(f"Failed to parse molecule in {file_path}")
                    return None

                try:
                    Chem.SanitizeMol(mol)
                except Exception as e:
                    logger.warning(f"Sanitization failed for {file_path}: {e}")
                    return None

                if mol.GetNumAtoms() == 0:
                    logger.warning(f"Empty molecule after load in {file_path}")
                    return None

                return mol

            except Exception as e:
                logger.warning(f"Error loading molecule from {file_path}: {e}")
                return None

    logger.warning(f"No ligand file found in {folder_path}")
    return None


def is_rdkit_friendly(folder_path: str) -> bool:
    mol = load_ligand_mol(folder_path)
    if mol is None:
        logger.info(f"Skipping {folder_path}: invalid or unparsable molecule")
        return False
    logger.info(f"Accepted {folder_path}: RDKit-parsable molecule")
    return True


def filter_rdkit_friendly_folders(config: Dict[str, str]) -> None:
    """
    Filters ligand folders under input_path.
    Copies only folders with RDKit-friendly ligand files to output_path.

    Args:
        config dict with keys:
            - "input_path": str, root folder containing ligand subfolders
            - "output_path": str, folder to copy RDKit-compatible folders into
    """
    input_path = config.get("input_path")
    output_path = config.get("output_path")

    if not input_path or not output_path:
        logger.error("Both 'input_path' and 'output_path' must be specified in config")
        return

    if not os.path.isdir(input_path):
        logger.error(f"Input path does not exist or is not a directory: {input_path}")
        return

    if not os.path.exists(output_path):
        os.makedirs(output_path)
        logger.info(f"Created output directory: {output_path}")

    copied_total = 0
    skipped_total = 0

    for folder_name in os.listdir(input_path):
        folder_path = os.path.join(input_path, folder_name)
        if os.path.isdir(folder_path):
            if is_rdkit_friendly(folder_path):
                dest_path = os.path.join(output_path, folder_name)
                try:
                    shutil.copytree(folder_path, dest_path)
                    logger.info(f"Copied folder: {folder_path} -> {dest_path}")
                    copied_total += 1
                except Exception as e:
                    logger.error(f"Failed to copy folder {folder_path} to {dest_path}: {e}")
            else:
                skipped_total += 1

    logger.info(f"Filtering complete. Total copied: {copied_total}, total skipped: {skipped_total}")
