import csv
from pathlib import Path

def create_project_status_csv(base_path):
    base_path = Path(base_path).resolve()
    output_csv = base_path / "project_file_status.csv"

    # Define allowed file extensions
    allowed_exts = {".txt", ".yaml", ".py", ".md"}

    # Exclusion rules
    def should_include(file_path):
        rel_path = file_path.relative_to(base_path)
        parts = rel_path.parts
        return (
            file_path.suffix in allowed_exts and
            "__init__.py" not in file_path.name and
            ".venv" not in parts and
            not rel_path.as_posix().startswith("models/backbone/IGModel/")
        )

    # Collect and filter files
    file_paths = [f for f in base_path.rglob("*") if f.is_file() and should_include(f)]
    if not file_paths:
        print("No matching files found in the specified directory.")
        return None

    # Prepare CSV
    headers = ["Relative Path", "Status", "Notes"]
    with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for fpath in file_paths:
            rel_path = fpath.relative_to(base_path)
            writer.writerow([str(rel_path), "", ""])

    print(f"CSV saved to: {output_csv}")
    return output_csv

# Example usage
if __name__ == "__main__":
    create_project_status_csv("/media/racah/2b2b05ab-497e-47ab-a698-6e77a3b775c4/grisha/for_ProtLigMap")
