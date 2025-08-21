import os
import shutil
from loguru import logger
from rdkit import Chem
from rdkit.Chem import Descriptors
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

RDLogger.logger().setLevel(RDLogger.WARNING)
import sys
sys.stderr = RDKitLogHandler()


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


def count_peptide_bonds(mol) -> int:
    count = 0
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtom()
        end = bond.GetEndAtom()
        if {begin.GetSymbol(), end.GetSymbol()} == {'C', 'N'}:
            c_atom = begin if begin.GetSymbol() == 'C' else end
            has_carbonyl = any(
                nbr.GetSymbol() == 'O' and mol.GetBondBetweenAtoms(c_atom.GetIdx(), nbr.GetIdx()).GetBondType() == Chem.rdchem.BondType.DOUBLE
                for nbr in c_atom.GetNeighbors()
            )
            if has_carbonyl:
                count += 1
    return count


def contains_alpha_carbon_pattern(mol) -> bool:
    pattern = Chem.MolFromSmarts('C([N])C=O')
    return mol.HasSubstructMatch(pattern)


def is_probably_peptide(mol) -> bool:
    peptide_bonds = count_peptide_bonds(mol)
    mw = Descriptors.MolWt(mol)

    if peptide_bonds > 10:
        logger.info(f"Rejected: peptide_bond_count = {peptide_bonds}")
        return True
    if mw > 2000:
        logger.info(f"Rejected: molecular weight = {mw:.1f} > 2000")
        return True
    if contains_alpha_carbon_pattern(mol):
        logger.info("Rejected: contains alpha-carbon motif")
        return True

    return False


def process_ligand_folder(folder_path: str) -> bool:
    mol = load_ligand_mol(folder_path)
    if mol is None:
        logger.info(f"Skipping {folder_path}: invalid molecule")
        return False

    if is_probably_peptide(mol):
        logger.info(f"Skipping {folder_path}: likely peptide ligand")
        return False

    logger.info(f"Keeping {folder_path}: passes filters")
    return True


def filter_ligand_folders(config: Dict[str, str]) -> None:
    """
    Processes ligand folders under input_path.
    Copies only *non-peptide* folders to output_path.

    Args:
        config dict with keys:
            - "input_path": str, root folder containing ligand subfolders
            - "output_path": str, folder to copy folders that are NOT peptides
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
    kept_total = 0

    for folder_name in os.listdir(input_path):
        folder_path = os.path.join(input_path, folder_name)
        if not os.path.isdir(folder_path):
            continue

        mol = load_ligand_mol(folder_path)
        if mol is None:
            logger.info(f"Skipping {folder_path}: invalid molecule")
            continue

        if not is_probably_peptide(mol):  # <--- Copy only if NOT a peptide
            dest_path = os.path.join(output_path, folder_name)
            try:
                shutil.copytree(folder_path, dest_path)
                logger.info(f"Copied (non-peptide): {folder_path} -> {dest_path}")
                copied_total += 1
            except Exception as e:
                logger.error(f"Failed to copie {folder_path} to {dest_path}: {e}")
        else:
            logger.info(f"Keeping (peptide): {folder_path}")
            kept_total += 1

    logger.info(f"Filtering complete. Total copied (non-peptides): {copied_total}, total kept (peptides): {kept_total}")

