import os
import shutil
from tqdm import tqdm
from typing import List, Tuple, Dict, Set
from src.utils.file_system import check_directory_exists
from loguru import logger



def copy_from_source(
    source_path: str,
    dest: str,
    file_count: int,
    folder_count: int,
    log: List[str]
) -> Tuple[int, int, List[str]]:
    """
    Copies contents from a single source folder to the destination.

    Parameters:
        source_path (str): Path to the source folder.
        dest (str): Destination folder path.
        file_count (int): Current total file count.
        folder_count (int): Current total folder count.
        log (List[str]): List to append copy logs to.

    Returns:
        Tuple[int, int, List[str]]: Updated file count, folder count, and log.
    """
    items = list(os.walk(source_path))
    all_paths = []

    for root, dirs, files in items:
        for d in dirs:
            all_paths.append(os.path.join(root, d))
        for f in files:
            all_paths.append(os.path.join(root, f))

    for path in tqdm(all_paths, desc=f"Copying from {os.path.basename(source_path)}"):
        rel_path = os.path.relpath(path, source_path)
        dest_path = os.path.join(dest, rel_path)

        if os.path.isdir(path):
            os.makedirs(dest_path, exist_ok=True)
            folder_count += 1
            msg = f"DIR  : {path} -> {dest_path}"
            log.append(msg)
            logger.info(msg)
        else:
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copy2(path, dest_path)
            file_count += 1
            msg = f"FILE : {path} -> {dest_path}"
            log.append(msg)
            logger.info(msg)

    return file_count, folder_count, log


def merge_folders_with_log(
    input_dict: Dict[str, str]
) -> List[str]:
    """
    Merges contents from multiple source folders into a single destination.

    Parameters:
        input_dict (Dict[str, str]):
            A dictionary with:
                - 'input_paths' (str): List of folder paths to merge, seperated by comma between them
                - 'output_path' (str): Destination folder path

    Returns:
        List[str]: A log of copy operations performed.
    """
    sources_comma_seperated: List[str] = input_dict["input_paths"]
    sources = [s.strip() for s in sources_comma_seperated.split(',')]

    dest: str = input_dict["output_path"]

    if not os.path.exists(dest):
        os.makedirs(dest)
        logger.info(f"Created destination folder: {dest}")

    log: List[str] = []
    file_count: int = 0
    folder_count: int = 0

    for src in sources:
        check_directory_exists(src)
        file_count, folder_count, log = copy_from_source(src, dest, file_count, folder_count, log)

    logger.info("\n Merge Summary:")
    logger.info(f"Total files copied   : {file_count}")
    logger.info(f"Total folders copied : {folder_count}")
    logger.info(f"Destination folder   : {dest}")

    return log


def read_pdb_codes(file_path: str) -> Set[str]:
    """Read PDB codes (lowercase) from a text file, ignoring empty lines."""
    codes = set()
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                codes.add(line.lower())
    logger.info(f"Read {len(codes)} entries from {file_path}")
    return codes


def get_pdb_folders(main_data_path: str) -> Set[str]:
    """Get all folder names (lowercase) in the main data directory."""
    items = os.listdir(main_data_path)
    dirs = {d.lower() for d in items if os.path.isdir(os.path.join(main_data_path, d))}
    logger.info(f"Found {len(dirs)} folders in main data path: {main_data_path}")
    return dirs


def exclude_intersection(config: Dict[str, str]) -> None:
    """
    Copies folders from main_data_path to output_path excluding those listed in exclude_folders_index_path.

    Args:
        config dict with keys:
            - 'exclude_folders_index_path': path to pdb txt file (set B)
            - 'main_data_path': path to folder containing pdb folders (set A)
            - 'output_path': destination folder path to copy filtered folders into
    """
    exclude_path = config["exclude_folders_index_path"]
    main_data_path = config["main_data_path"]
    output_path = config["output_path"]

    exclude_set = read_pdb_codes(exclude_path)
    main_folders = get_pdb_folders(main_data_path)

    # Folders to copy = main_folders - exclude_set
    to_copy = main_folders - exclude_set

    logger.info(f"Total folders in source: {len(main_folders)}")
    logger.info(f"Folders to exclude: {len(exclude_set)}")
    logger.info(f"Folders to copy (A \\ B): {len(to_copy)}")

    if not os.path.exists(output_path):
        os.makedirs(output_path)
        logger.info(f"Created output folder: {output_path}")

    for pdb_id in to_copy:
        src_folder = os.path.join(main_data_path, pdb_id)
        dest_folder = os.path.join(output_path, pdb_id)
        if os.path.isdir(src_folder):
            try:
                shutil.copytree(src_folder, dest_folder)
                logger.info(f"Copied {src_folder} -> {dest_folder}")
            except FileExistsError:
                logger.warning(f"Destination folder already exists, skipping: {dest_folder}")
            except Exception as e:
                logger.error(f"Error copying {src_folder} to {dest_folder}: {e}")
        else:
            logger.warning(f"Source folder does not exist (skipped): {src_folder}")

    logger.info("Copy operation complete.")

