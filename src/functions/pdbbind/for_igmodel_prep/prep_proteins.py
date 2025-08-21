import os
from pathlib import Path
from Bio import PDB
from typing import Dict
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.utils.env_utils import setup_mgltools
from src.utils.file_system import check_directory_exists
from src.utils.subprocess_utils import safe_run
from loguru import logger


def clean_pdb_remove_hetatoms(input_pdb_path, output_pdb_path):
    # Cleaning protein files from Hydrogens and non-amino-acid residues
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', str(input_pdb_path))

    class CleanSelect(PDB.Select):
        # Cleaning atoms which are hydrogens for consistency
        def accept_atom(self, atom):
            return atom.element != 'H' # Skip hydrogens

        def accept_residue(self, residue):
            # Here we only want to keep the amino acids which are the essense of the protein structure
            # It's mainly done to make it easy for different programs to work with dockings
            standard_aas = {
                'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY',
                'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER',
                'THR', 'TRP', 'TYR', 'VAL'
            }
            return residue.get_resname() in standard_aas

    io = PDB.PDBIO()
    io.set_structure(structure)
    # In this part we can input a class that passes PDB.Select inside, and can alter 4 types of functions: 
    # accept_model, accept_chain, accept_residue, accept_atom by our own filter accoridngly.
    io.save(str(output_pdb_path), CleanSelect())

def fix_pdb_element_column(pdb_path: Path) -> None:
    """
    Fixes the element column in a PDB file by inferring the element from atom names.

    Args:
        pdb_path (Path): Path to the PDB file to be fixed.
    """
    element_mapping = {
        "C": "C", "CA": "C", "CB": "C", "CD": "C", "CG": "C", "CE": "C", "CZ": "C",
        "N": "N", "ND": "N", "NE": "N", "NH": "N",
        "O": "O", "OD": "O", "OE": "O", "OG": "O", "OH": "O",
        "S": "S", "SD": "S", "SG": "S",
        "H": "H", "HA": "H", "HB": "H", "HG": "H", "HD": "H",
    }

    def extract_element(atom_name: str) -> str:
        key = ''.join(filter(str.isalpha, atom_name)).upper()
        return element_mapping.get(key, 'C')  # Default to carbon

    fixed_lines = []
    with open(pdb_path, 'r') as file:
        for line in file:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                atom_name = line[12:16].strip()
                element = extract_element(atom_name).rjust(2)
                line = line[:76] + element + line[78:]
            fixed_lines.append(line)

    with open(pdb_path, 'w') as file:
        file.writelines(fixed_lines)


def remove_oxt_atoms(pdb_path):
    with open(pdb_path, "r") as f:
        lines = f.readlines()
    cleaned_lines = [line for line in lines if not (" OXT " in line and (line.startswith("ATOM") or line.startswith("HETATM")))]
    with open(pdb_path, "w") as f:
        f.writelines(cleaned_lines)

def validate_protein_pdbqt(pdbqt_path):
    if not os.path.isfile(pdbqt_path) or os.path.getsize(pdbqt_path) == 0:
        logger.error(f"PDBQT file does not exist or is empty: {pdbqt_path}")
        return False

    with open(pdbqt_path, 'r') as f:
        lines = f.readlines()

    if not any("TORSDOF" in line for line in lines):
        logger.info(f"Adding missing TORSDOF line to {pdbqt_path}")
        with open(pdbqt_path, 'a') as f_append:
            f_append.write("\nTORSDOF 0\n")

    atom_lines = [line for line in lines if line.startswith("ATOM")]
    if not atom_lines:
        logger.error(f"No ATOM entries found in {pdbqt_path}")
        return False

    return True

def remove_torsdof(pdbqt_path):
    with open(pdbqt_path, 'r') as f:
        lines = f.readlines()
    cleaned = [line for line in lines if not line.strip().startswith("TORSDOF")]
    with open(pdbqt_path, 'w') as f:
        f.writelines(cleaned)

def process_protein_complex(complex_dir: Path, output_dir: Path, MGLPYTHON: str, PREPARE_RECEPTOR: str) -> bool:
    complex_id = complex_dir.name
    complex_output_dir = output_dir / complex_id
    complex_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Starting processing for: {complex_id}")

    protein = complex_dir / f"{complex_id}_protein.pdb"
    cleaned_pdb = complex_output_dir / f"{complex_id}_protein_cleaned.pdb"
    prepared_pdb = complex_output_dir / f"{complex_id}_protein_prepared.pdb"
    protein_pdbqt = complex_output_dir / f"{complex_id}_protein.pdbqt"

    if not protein.exists():
        logger.warning(f"Protein file missing: {protein}, skipping...")
        return False

    if not cleaned_pdb.exists() or cleaned_pdb.stat().st_size == 0:
        # Step 1 - Initial cleaning of the PDB protein file (remove HETATM & non-standard AAs)
        logger.info(f"Cleaning protein PDB for {complex_id}")
        clean_pdb_remove_hetatoms(protein, cleaned_pdb)

    # Step 2 - Fix element column (ensures element info is present in the PDB)
    logger.info(f"Fixing element column in cleaned PDB for {complex_id}")
    fix_pdb_element_column(cleaned_pdb)

    if not prepared_pdb.exists() or prepared_pdb.stat().st_size == 0:
        # Step 3 - Convert cleaned PDB to a format suitable for docking using MGLTools
        logger.info(f"Preparing protein PDB for {complex_id}")
        if not safe_run([MGLPYTHON, PREPARE_RECEPTOR, "-r", str(cleaned_pdb), "-o", str(prepared_pdb)],
                        f"Preparing cleaned PDB for {complex_id}"):
            return False

    # Step 4 - Remove OXT atoms (must be done *after* MGLTools conversion, not before)
    logger.info(f"Removing OXT atoms for {complex_id}")
    remove_oxt_atoms(prepared_pdb)

    # Step 5 - Fix element column again (MGLTools may introduce weird element values)
    logger.info(f"Final cleanup of element column in prepared PDB for {complex_id}")
    fix_pdb_element_column(prepared_pdb)

    # Step 6 - Generate the final .pdbqt file for docking
    if not protein_pdbqt.exists() or protein_pdbqt.stat().st_size == 0:
        if not safe_run([MGLPYTHON, PREPARE_RECEPTOR, "-r", str(prepared_pdb), "-o", str(protein_pdbqt)],
                        f"Preparing PDBQT for {complex_id}"):
            return False

    # Step 7 - Validate the PDBQT format to ensure docking compatibility
    if not validate_protein_pdbqt(protein_pdbqt):
        return False

    # Step 8 - Remove TORSDOF field (torsion degrees of freedom, only needed for ligands)
    remove_torsdof(protein_pdbqt)
    logger.info(f"Done with {complex_id}")
    return True



def process_all_protein_complexes(config: Dict[str, str]) -> None:
    """
    Process all protein complexes from their raw PDB form.
    Cleans and prepares the proteins for docking, following preprocessing steps from the igmodel paper.

    Args:
        config (dict): Configuration dictionary containing:
            - "input_path": str, path to protein complexes
            - "output_path": str, path to write cleaned complexes
            - "max_workers": int, number of workers for parallelism
            - "mgtools_path": str, root path to MGLTools
    """
    # Setup environment for MGLTools
    mgl_root, MGLPYTHON, PREPARE_RECEPTOR, _ = setup_mgltools(config["mgtools_path"])

    input_path = check_directory_exists(config.get("input_path"))
    output_path = check_directory_exists(config.get("output_path"))
    number_of_workers = int(config.get("max_workers", 1))

    logger.info(f"Processing dataset: {input_path}")
    complex_dirs = [d for d in Path(input_path).iterdir() if d.is_dir()]
    logger.info(f"Found {len(complex_dirs)} protein complexes")

    # Run processing for each protein complex in parallel
    with ProcessPoolExecutor(max_workers=number_of_workers) as executor:
        futures = {
            executor.submit(process_protein_complex, d, output_path, MGLPYTHON, PREPARE_RECEPTOR): d
            for d in complex_dirs
        }

        for future in as_completed(futures):
            complex_path = futures[future]
            try:
                result = future.result()
                if not result:
                    logger.error(f"Processing failed for: {complex_path.name}")
            except Exception as e:
                logger.error(f"Exception while processing {complex_path.name}: {e}")


logger.info("Processing completed successfully.")