import sys
import subprocess

def save_installed_packages_to_requirements():
    # Get the current Python interpreter path
    python_interpreter = sys.executable
    print(f"Using Python interpreter: {python_interpreter}")

    try:
        # Use pip to get list of installed packages
        result = subprocess.run(
            [python_interpreter, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Write to requirements.txt
        with open("requirements.txt", "w") as f:
            f.write(result.stdout)
        
        print("requirements.txt has been created successfully.")
    
    except subprocess.CalledProcessError as e:
        print("Failed to generate requirements.txt")
        print("Error:", e.stderr)

if __name__ == "__main__":
    save_installed_packages_to_requirements()
