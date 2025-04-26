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

def create_activation_command() -> list[str]:
    activation_command = []    
    if os.name == "nt":
        activation_script = os.path.join(VENV_NAME, "Scripts", "activate.bat")
        activation_command = ["cmd.exe", "/k", activation_script]
    else:
        bash_command = f"source {VENV_NAME}/bin/activate && exec $SHELL"
        activation_command = ["bash", "-c", bash_command]
    return activation_command

def main():
    root_path = os.path.abspath(os.path.dirname(__file__))
    create_virtual_environment()
    add_root_to_sys_path(root_path)
    install_requirements()
    activation_command = create_activation_command()
    input("Virtual environment setup complete. Press Enter to finish and activate environment.")
    subprocess.run(activation_command, shell=True)
    
    sys.exit(0)

if __name__ == "__main__":
    main()
