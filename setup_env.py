import os
import subprocess
import sys

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(PROJECT_DIR, ".venv")
PYTHON_GLOBAL_EXEC = sys.executable
PYTHON_VENV_EXEC = (
    os.path.join(VENV_DIR, "Scripts", "python.exe")
    if os.name == "nt"
    else os.path.join(VENV_DIR, "bin", "python")
)
HCMSFEM_DIR = os.path.join(PROJECT_DIR, "hcmsfem")


def install_hcmsfem():
    subprocess.check_call(
        [
            PYTHON_GLOBAL_EXEC,
            os.path.join(HCMSFEM_DIR, "setup_env.py"),
            "--parent",
            "--no-activate",
        ]
    )
    print(f"✅ Installed hcmsfem package in virtual environment {VENV_DIR}\n")


def install_requirements():
    subprocess.check_call(
        [PYTHON_VENV_EXEC, "-m", "pip", "install", "-r", "requirements.txt"]
    )
    print(f"✅ Installed requirements from {os.path.join(HCMSFEM_DIR, 'requirements.txt')}\n")


def activate_environment():
    print()
    subprocess.run(
        [PYTHON_VENV_EXEC, os.path.join(HCMSFEM_DIR, "activate_env.py")],
        shell=True,
        check=True,
    )


if __name__ == "__main__":
    install_hcmsfem()
    install_requirements()
    activate_environment()
