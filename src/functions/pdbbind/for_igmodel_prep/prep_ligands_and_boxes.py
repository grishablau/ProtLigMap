from pathlib import Path
from loguru import logger
import numpy as np
from typing import Dict, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.utils.env_utils import setup_mgltools
from src.utils.file_system import check_directory_exists
from src.utils.subprocess_utils import safe_run


def get_ligand_coords(filepath: Path) -> np.ndarray:
    coords = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')):
                try:
                    coords.append([
                        float(line[30:38].strip()),
                        float(line[38:46].strip()),
                        float(line[46:54].strip())
                    ])
                except ValueError:
                    continue
    return np.array(coords)


def compute_box(coords: np.ndarray, padding=5.0):
    min_xyz = coords.min(axis=0)
    max_xyz = coords.max(axis=0)
    center = (min_xyz + max_xyz) / 2
    size = (max_xyz - min_xyz) + padding
    return center, size


def convert_ligand_to_pdbqt(
    input_folder: Path,
    output_folder: Path,
    ligand_id: str,
    MGLPYTHON: str,
    PREPARE_LIGAND: str
) -> Optional[str]:
    input_mol2 = input_folder / f"{ligand_id}_ligand.mol2"
    input_sdf = input_folder / f"{ligand_id}_ligand.sdf"
    ligand_input = input_mol2 if input_mol2.exists() else input_sdf if input_sdf.exists() else None

    if ligand_input is None:
        logger.warning(f"No ligand file found for {ligand_id} in {input_folder}")
        return None

    output_folder.mkdir(parents=True, exist_ok=True)

    fixed_path = output_folder / f"{ligand_id}_ligand_fixed.mol2"
    pdbqt_path = output_folder / f"{ligand_id}_ligand.pdbqt"

    if pdbqt_path.exists():
        return str(pdbqt_path)

    obabel_cmd = [
        "obabel", "-i", ligand_input.suffix[1:], str(ligand_input), "-o", "mol2",
        "-O", str(fixed_path), "-h"
    ]

    if not safe_run(obabel_cmd, f"Adding hydrogens to {ligand_input.name}", cwd=str(output_folder)):
        return None

    ligand_cmd = [str(MGLPYTHON), str(PREPARE_LIGAND), "-l", fixed_path.name, "-o", pdbqt_path.name]

    logger.info(f"Running ligand_cmd: {' '.join(ligand_cmd)} in {output_folder}")


    success, output = safe_run(
        ligand_cmd,
        f"Converting ligand {ligand_id} to PDBQT",
        cwd=str(output_folder)
    )

    if not success:
        logger.error(f"Conversion failed for {ligand_id}. Error:\n{output}")
        return None


    logger.info(f"Successfully converted ligand {ligand_id} to .pdbqt")
    return str(pdbqt_path)


def process_ligand_folder(folder: str, input_dir: Path, output_dir: Path, MGLPYTHON: str, PREPARE_LIGAND: str):
    input_path = input_dir / folder
    output_path = output_dir / folder

    if not input_path.is_dir():
        return None

    ligand_file = output_path / f"{folder}_ligand.pdbqt"
    if not ligand_file.exists():
        ligand_file = convert_ligand_to_pdbqt(input_path, output_path, folder, MGLPYTHON, PREPARE_LIGAND)
        if ligand_file is None:
            return None

    coords = get_ligand_coords(ligand_file)
    if coords.size == 0:
        logger.warning(f"No coordinates found in {ligand_file}")
        return None

    center, size = compute_box(coords)
    return {
        "pdb_id": folder,
        "center_x": center[0],
        "center_y": center[1],
        "center_z": center[2],
        "size_x": size[0],
        "size_y": size[1],
        "size_z": size[2],
    }


def process_all_ligands(config: Dict[str, str]) -> None:
    """
    Process all ligand files to prepare them for docking.
    This includes converting ligands to .pdbqt format and computing box dimensions around the ligand.

    Steps:
    1. Detect the ligand file (.mol2 or .sdf) in each folder.
    2. Add explicit hydrogens using Open Babel, as docking tools like AutoDock require them for:
       - hydrogen bond modeling
       - accurate partial charge assignment
       - correct rotatable bond detection
    3. Convert the hydrogenated ligand to .pdbqt format using MGLTools' prepare_ligand4.py.
    4. Extract 3D coordinates and compute center and size of the docking box.

    Args:
        config (dict): Configuration dictionary containing:
            - "input_path": str, path to input folders each containing a ligand file
            - "output_path": str, folder to write .pdbqt files
            - "max_workers": int, number of parallel workers
            - "mgtools_path": str, path to MGLTools root folder
            - "docking_path": str, path to write the docking txt file
    """

    _, MGLPYTHON, _, PREPARE_LIGAND = setup_mgltools(config["mgtools_path"])
    input_dir = check_directory_exists(config["input_path"])
    output_dir = check_directory_exists(config["output_path"])
    number_of_workers = int(config.get("max_workers", 1))

    docking_path = check_directory_exists(config["docking_path"])
    folders = [f.name for f in input_dir.iterdir() if f.is_dir()]
    results = []

    with ProcessPoolExecutor(max_workers=number_of_workers) as executor:
        futures = {
            executor.submit(process_ligand_folder, f, input_dir, output_dir, MGLPYTHON, PREPARE_LIGAND): f
            for f in folders
        }
        for future in as_completed(futures):
            res = future.result()
            if res:
                results.append(res)

    docking_file_path = docking_path / "docking_box_coordinates_exp.txt"
    with open(docking_file_path, "w") as f:
        f.write("pdb_id\tcenter_x\tcenter_y\tcenter_z\tsize_x\tsize_y\tsize_z\n")
        for r in results:
            f.write(f"{r['pdb_id']}\t{r['center_x']:.3f}\t{r['center_y']:.3f}\t{r['center_z']:.3f}\t"
                    f"{r['size_x']:.3f}\t{r['size_y']:.3f}\t{r['size_z']:.3f}\n")

    logger.info(f"Processed {len(results)} ligand files. Output saved to {docking_file_path}")


