# utils/param_choice_utils/__init__.py
import os
import glob
import importlib
import inspect

__all__ = []

current_dir = os.path.dirname(__file__)

# Find all .py files in this folder, except __init__.py
py_files = glob.glob(os.path.join(current_dir, "*.py"))
for py_file in py_files:
    module_name = os.path.basename(py_file)[:-3]
    if module_name == "__init__":
        continue

    # Import the module dynamically
    module = importlib.import_module(f".{module_name}", package=__name__)

    # Import all functions only
    for name, obj in inspect.getmembers(module, inspect.isfunction):
        globals()[name] = obj
        __all__.append(name)
