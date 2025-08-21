import sys
from pathlib import Path
from loguru import logger
import re

# Set up loguru logger to write logs to a file in the current directory
logger.remove()  # Remove default logger
log_file_path = Path(__file__).parent / "log_analysis_output.log"
logger.add(log_file_path, level="INFO", format="{time} | {level} | {message}")

# Regex pattern to extract log level from line
log_level_pattern = re.compile(r"\|\s*(DEBUG|INFO|WARNING|ERROR|CRITICAL)\s*\|")

def analyze_logs(folder_path: str):
    folder = Path(folder_path)
    if not folder.is_dir():
        logger.error(f"The provided path is not a folder: {folder}")
        return

    total_lines = 0
    log_level_counts = {}
    error_locations = []

    for file in folder.glob("*.log"):
        logger.info(f"Processing file: {file.name}")
        try:
            with file.open("r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, start=1):
                    total_lines += 1
                    match = log_level_pattern.search(line)
                    if match:
                        level = match.group(1)
                        log_level_counts[level] = log_level_counts.get(level, 0) + 1
                        if level == "ERROR":
                            error_locations.append((file.name, line_num))
        except Exception as e:
            logger.error(f"Failed to read {file.name}: {e}")

    logger.info(f"Total number of lines: {total_lines}")
    
    logger.info("Log level counts:")
    for level, count in log_level_counts.items():
        logger.info(f"  {level}: {count}")
    
    if error_locations:
        logger.warning("Errors found in the following files and lines:")
        for file_name, line_num in error_locations:
            logger.warning(f"  {file_name} at line {line_num}")
    else:
        logger.info("No ERROR lines found in any file.")

if __name__ == "__main__":
    analyze_logs("/media/racah/2b2b05ab-497e-47ab-a698-6e77a3b775c4/grisha/for_ProtLigMap/logs/run_2025-07-15_16-43-54")
