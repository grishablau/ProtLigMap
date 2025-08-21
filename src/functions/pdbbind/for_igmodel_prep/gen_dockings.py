from loguru import logger
import sys
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.utils.file_system import check_directory_exists, check_file_exists
from src.utils.vina_docking_runner import run_vina_docking
import os

def process_complex(
    complex_dir: Path,
    dataset_root: Path,
    box_file: Path,
    vina_path: Path
) -> Dict:
    complex_id = complex_dir.name
    logger.info(f"Docking for {complex_id}")

    try:
        result = run_vina_docking(
            pdb_id=complex_id,
            dataset_root=dataset_root,
            box_file=box_file,
            vina_path=vina_path
        )
        return {
            "complex_id": complex_id,
            "docking_success": result["success"],
            "warnings": "" if result["success"] else result["reason"]
        }
    except Exception as e:
        logger.error(f"Error during docking for {complex_id}: {e}")
        return {
            "complex_id": complex_id,
            "docking_success": False,
            "warnings": str(e)
        }


def run_batch_docking(config: Dict[str, str]) -> None:
    """
    Run docking for all complexes listed in the dataset folder.

    Args:
        config (dict): Configuration dictionary containing:
            - "max_workers": int, level of parallelism
            - "dataset_path": str, path to folder with complex subfolders
            - "docking_summaries_path": str, folder to save docking summaries
            - "box_file_path": str, path to box coordinate file
            - "vina_path": str, path to vina processing file
            
            
    """
    number_of_workers = int(config.get("max_workers", 1))
    dataset_path = check_directory_exists(config["dataset_path"])
    docking_summaries_path = check_directory_exists(config["docking_summaries_path"])
    box_file = check_file_exists(config["box_file_path"]) 
    



    vina_path = Path(config.get("vina_path")).expanduser()
    vina_path = check_file_exists(vina_path, abs_path_convert=False)

    complex_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
    if not complex_dirs:
        logger.warning("No complexes found to process.")
        return

    results = []
    with ProcessPoolExecutor(max_workers=number_of_workers) as executor:
        futures = {
            executor.submit(process_complex, d, dataset_path, box_file, vina_path): d.name            
            for d in complex_dirs
        } 
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                if not result["docking_success"]:
                    logger.error(f"Docking failed for {result['complex_id']}")
            except Exception as e:
                logger.error(f"Exception in subprocess: {e}")
                sys.exit(1)

    # Save results
    csv_path = docking_summaries_path / "docking_summary.csv"
    json_path = docking_summaries_path / "docking_summary.json"

    with open(csv_path, "w", newline="") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    with open(json_path, "w") as f_json:
        json.dump(results, f_json, indent=2)

    logger.info(f"Docking summary saved to: {csv_path} and {json_path}")
    
    
