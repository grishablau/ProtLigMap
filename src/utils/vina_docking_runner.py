import os
import subprocess
from loguru import logger
from pathlib import Path
from src.utils.file_system import find_subfolder, skip_if_condition


def validate_docking_output(pdbqt_path):
    if not os.path.isfile(pdbqt_path) or os.path.getsize(pdbqt_path) == 0:
        return False, "Docking output file is missing or empty."

    with open(pdbqt_path) as f:
        lines = f.readlines()

    model_blocks = [i for i, line in enumerate(lines) if line.strip().startswith("MODEL")]
    atom_blocks = [line for line in lines if line.strip().startswith("ATOM")]

    if not model_blocks:
        return False, "No MODEL blocks found — docking likely failed."
    if not atom_blocks:
        return False, "No ATOM lines found in docking output."

    score_lines = [line for line in lines if line.strip().startswith("REMARK VINA RESULT")]
    if score_lines:
        best_score = score_lines[0].split()[3]
        return True, f"Docking OK. Top Vina score: {best_score}"
    return True, "Docking OK. No score lines found."


# === Docking Execution ===
def run_vina_docking(
    pdb_id: str,
    dataset_root: Path,
    box_file: Path,
    vina_path: Path
) -> dict:
    """
    Runs docking for a single pdb_id. Returns a dictionary with success info.
    """
    
    vina_executable = os.path.expanduser(str(vina_path))
    dataset_root = Path(dataset_root)
    box_file = Path(box_file)

    with open(box_file, "r") as f:
        next(f)  # Skip header
        for line in f:
            row_pdb_id, cx, cy, cz, sx, sy, sz = line.strip().split("\t")

            if row_pdb_id != pdb_id:
                continue
            check_folder_or_file = "folder"
            folder_path = find_subfolder(dataset_root, pdb_id)
            if skip_if_condition(folder_path, pdb_id, "Folder not found.", logger, check="absence", check_folder_or_file=check_folder_or_file):
                error_message = f"{check_folder_or_file} not found"
                logger.error(error_message)
                return {"success": False, "reason": error_message}


            protein_path = folder_path / f"{pdb_id}_protein.pdbqt"
            ligand_path = folder_path / f"{pdb_id}_ligand.pdbqt"
            output_path = folder_path / f"{pdb_id}_docked.pdbqt"
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Skip conditions
            for path_check, desc in [
                (output_path, "Output already exists"),
                (protein_path, "Missing receptor file"),
                (ligand_path, "Missing ligand file")
            ]:
                if skip_if_condition(path_check, pdb_id, desc, logger, check="existence" if "exists" in desc else "absence", check_folder_or_file="file"):
                    logger.error(desc)
                    return {"success": False, "reason": desc}


            # Run vina
            cmd = [
                vina_executable,
                "--receptor", str(protein_path),
                "--ligand", str(ligand_path),
                "--center_x", cx,
                "--center_y", cy,
                "--center_z", cz,
                "--size_x", "15",
                "--size_y", "15",
                "--size_z", "15",
                "--exhaustiveness", "8",
                "--num_modes", "15",
                "--out", str(output_path)
            ]

            logger.info(f"[RUN] {pdb_id}: Docking started...")
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.stdout:
                logger.debug(result.stdout)
            if result.stderr:
                logger.warning(result.stderr)

            if result.returncode != 0:
                return {"success": False, "reason": f"Vina failed with code {result.returncode}"}

            valid, reason = validate_docking_output(output_path)
            return {
                "success": valid,
                "reason": reason if valid else f"Validation failed: {reason}"
            }

    return {"success": False, "reason": f"{pdb_id} not found in box file"}
