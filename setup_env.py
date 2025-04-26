import os
import sys
import subprocess
import venv
VENV_NAME = ".venv2"

def create_virtual_environment():
    if not os.path.exists(VENV_NAME):
        venv.create(VENV_NAME, with_pip=True)
        print(f"Created virtual environment '{VENV_NAME}'")
    else:
        print(f"Virtual environment '{VENV_NAME}' already exists")

def add_root_to_sys_path(root_path):
    sitecustomize_path = os.path.join(root_path, VENV_NAME, "Lib", "site-packages", "sitecustomize.py")
    if not os.path.exists(sitecustomize_path):
        os.makedirs(os.path.dirname(sitecustomize_path), exist_ok=True)
    with open(sitecustomize_path, "w") as f:
        f.write(f"import sys\nsys.path.insert(0, r'{root_path}')\n")
    print("Configured sys.path to include the repo root")

def install_requirements():
    python_exec = os.path.join(VENV_NAME, "Scripts", "python.exe") if os.name == "nt" else os.path.join(VENV_NAME, "bin", "python")
    
    subprocess.check_call([python_exec, "-m", "pip", "install", "--upgrade", "pip"])
    print("Upgraded pip to latest version")
    
    subprocess.check_call([python_exec, "-m", "pip", "install", "-r", "requirements.txt"])
    print("Installed required packages")

def activate_environment():
    if os.name == "nt":
        activation_script = os.path.join(VENV_NAME, "Scripts", "activate.bat")
        input("Press enter to activate the environment.")    
        subprocess.run(["cmd.exe", "/k", activation_script])
    else:
        activation_script = f"{VENV_NAME}/bin/activate"
        print("To activate the environment, run:")
        print(f"source {activation_script}")

def main():
    root_path = os.path.abspath(os.path.dirname(__file__))
    create_virtual_environment()
    add_root_to_sys_path(root_path)
    install_requirements()
    activate_environment()
    sys.exit(0)

if __name__ == "__main__":
    main()
