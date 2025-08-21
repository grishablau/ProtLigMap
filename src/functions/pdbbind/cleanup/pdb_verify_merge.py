import os
from loguru import logger
from typing import Set, Dict



def get_pdb_codes_from_index(file_path: str) -> Set[str]:
    """Parse PDB codes from the index file (case-insensitive)."""
    codes = set()
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            pdb_code = line.split()[0].lower()
            codes.add(pdb_code)
    logger.info(f"Loaded {len(codes)} entries from index file {file_path}")
    return codes


def get_folders_in_folder(folder_path: str) -> Set[str]:
    """Return set of folder names (lowercase) inside the folder."""
    items = os.listdir(folder_path)
    dirs = {d.lower() for d in items if os.path.isdir(os.path.join(folder_path, d))}
    logger.info(f"Found {len(dirs)} folders in folder {folder_path}")
    return dirs


def _ask_continue(question: str) -> bool:
    """Helper to ask user Y/N question, return True if yes."""
    while True:
        answer = input(f"\n {question} (y/n): ").strip().lower()
        if answer in {'y', 'yes'}:
            return True
        elif answer in {'n', 'no'}:
            return False
        else:
            print("Please answer 'y' or 'n'.")


def verify_index_vs_folder(config: Dict[str, str]) -> None:
    """
    Verifies that folders inside the given folder exactly match the entries in the index file.
    Logs any mismatches found.
    Optionally asks for user confirmation if mismatches exist.

    Args:
        config (Dict[str, str]) with keys:
            - 'metadata_path': full path to index file
            - 'data_path': full path to folder containing subfolders
            - 'require_confirmation': 'TRUE' or 'FALSE' (case-insensitive, default 'FALSE')
    """
    index_path = config["metadata_path"]
    folder = config["data_path"]
    require_confirmation = config.get("require_confirmation", "FALSE").upper()

    pdb_codes = get_pdb_codes_from_index(index_path)
    folder_dirs = get_folders_in_folder(folder)

    missing_folders = pdb_codes - folder_dirs
    extra_folders = folder_dirs - pdb_codes

    logger.info(f"Total index entries: {len(pdb_codes)}")
    logger.info(f"Total folders in folder: {len(folder_dirs)}")

    if missing_folders:
        logger.warning(f"Index entries WITHOUT matching folders: {len(missing_folders)}")
        for code in sorted(missing_folders):
            logger.warning(f"  - {code}")
    else:
        logger.info("All index entries have matching folders.")

    if extra_folders:
        logger.warning(f"Folders WITHOUT matching index entries: {len(extra_folders)}")
        for folder_name in sorted(extra_folders):
            logger.warning(f"  - {folder_name}")
    else:
        logger.info("All folders have matching index entries.")

    if require_confirmation == "TRUE" and (missing_folders or extra_folders):
        if not _ask_continue("Mismatches detected. Do you want to continue?"):
            logger.info(" Operation aborted by user.")
            exit(1)

    logger.info("Verification complete.")
