import os
import subprocess
import sys
import json

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

def merge_dicts(a, b):
    """Merge dict b into dict a, appending lists and updating keys."""
    for key, value in b.items():
        if key in a:
            if isinstance(a[key], list) and isinstance(value, list):
                a[key].extend(value)
            elif isinstance(a[key], dict) and isinstance(value, dict):
                merge_dicts(a[key], value)
            else:
                a[key] = value  # Overwrite for non-list/dict
        else:
            a[key] = value
    return a

def setup_vscode_configs():
    vscode_dir = os.path.join(PROJECT_DIR, ".vscode")
    os.makedirs(vscode_dir, exist_ok=True)
    settings_src = os.path.join(PROJECT_DIR, "vscode_settings.json")
    settings_dest = os.path.join(vscode_dir, "settings.json")
    tasks_dest = os.path.join(vscode_dir, "tasks.json")

    # Load source config
    with open(settings_src, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Merge settings
    settings = {}
    if os.path.exists(settings_dest):
        with open(settings_dest, "r", encoding="utf-8") as f:
            try:
                settings = json.load(f)
            except Exception:
                settings = {}
    if "settings" in config:
        settings = merge_dicts(settings, config["settings"])
        with open(settings_dest, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=4)
        print(f"✅ Merged VSCode settings at {settings_dest}")

    # Merge tasks
    tasks = {}
    if os.path.exists(tasks_dest):
        with open(tasks_dest, "r", encoding="utf-8") as f:
            try:
                tasks = json.load(f)
            except Exception:
                tasks = {}
    if "tasks" in config:
        # For tasks, merge top-level keys and append to "tasks" list
        for key in config["tasks"]:
            if key == "tasks":
                if "tasks" not in tasks or not isinstance(tasks["tasks"], list):
                    tasks["tasks"] = []
                tasks["tasks"].extend(config["tasks"]["tasks"])
            else:
                tasks[key] = config["tasks"][key]
        with open(tasks_dest, "w", encoding="utf-8") as f:
            json.dump(tasks, f, indent=4)
        print(f"✅ Merged VSCode tasks at {tasks_dest}")

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
    setup_vscode_configs()
    activate_environment()
