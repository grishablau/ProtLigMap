#

import sys
import os

# Desired interpreter path
DESIRED_PYTHON = "/media/racah/2b2b05ab-497e-47ab-a698-6e77a3b775c4/grisha/for_ProtLigMap/.venv/bin/python"

if sys.executable != DESIRED_PYTHON:
    os.execv(DESIRED_PYTHON, [DESIRED_PYTHON] + sys.argv)


# For debugging


import sys

print("\nPython Interpreter:", sys.executable)

from pathlib import Path
from src.entrypoints.setup_logger_and_root import setup_logger_and_root
from src.entrypoints.initialise import initialise
from src.entrypoints.run_pipeline import run_pipeline

# Setup logger using the location of this script
ROOT = setup_logger_and_root(Path(__file__).resolve())

def main():
    pipeline = initialise()
    run_pipeline(pipeline)

if __name__ == "__main__":
    main()
