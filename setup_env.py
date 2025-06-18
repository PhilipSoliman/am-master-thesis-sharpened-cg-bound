import os
import subprocess
import sys
import venv

VENV_NAME = ".venv"
DYNAMIC_LIBRARY_FOLDER = "clibs"
PYTHON_EXEC = (
    os.path.join(VENV_NAME, "Scripts", "python.exe")
    if os.name == "nt"
    else os.path.join(VENV_NAME, "bin", "python")
)


def create_virtual_environment():
    if not os.path.exists(VENV_NAME):
        venv.create(VENV_NAME, with_pip=True)
        print(f"Created virtual environment '{VENV_NAME}'")
    else:
        print(f"Virtual environment '{VENV_NAME}' already exists")


def install_requirements(root_path):

    # upgrade pip
    subprocess.check_call([PYTHON_EXEC, "-m", "pip", "install", "--upgrade", "pip"])
    print("Upgraded pip to latest version")

    # Install requirements from requirements.txt
    subprocess.check_call(
        [
            PYTHON_EXEC,
            "-m",
            "pip",
            "install",
            "-r",
            os.path.join(root_path, "requirements.txt"),
        ]
    )
    print("Installed required packages")

    # install PyTorch
    if os.name == "nt":
        torch_url = "https://download.pytorch.org/whl/cu126"
        subprocess.check_call(
            [
                PYTHON_EXEC,
                "-m",
                "pip",
                "install",
                "torch",
                "--index-url",
                torch_url,
            ]
        )
    else:
        subprocess.check_call(
            [
                PYTHON_EXEC,
                "-m",
                "pip",
                "install",
                "torch",
            ]
        )
    print("Installed PyTorch")

    subprocess.check_call([PYTHON_EXEC, "-m", "pip", "install", "-e", "."])
    print("Installed local packages")


def generate_meshes_for_experiments(root_path):
    mesh_script = os.path.join(root_path, "generate_meshes.py")
    print(f"Generating meshes for experiments using {mesh_script}...")

    result = subprocess.run(
        [PYTHON_EXEC, mesh_script],
        cwd=root_path,
        # stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    print(result.stdout)

    if result.returncode != 0:
        print("Failed to generate meshes. Exiting.")
        sys.exit(1)

    print("Meshes generated successfully.")


def make_c_library(root_path):
    lib_path = os.path.join(root_path, "lib", "solvers", "clib")
    print(f"Building C library in {lib_path}...")

    result = subprocess.run(
        ["make"],
        cwd=lib_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        shell=True,  # Needed for Windows compatibility with make
    )

    print(result.stdout)

    if result.returncode != 0:
        print("Failed to build the C library. Exiting.")
        sys.exit(1)

    print("C library built successfully.")


def activate_environment():
    if os.name == "nt":
        activation_script = os.path.join(VENV_NAME, "Scripts", "activate.bat")
        input("Press Enter to activate the environment.")
        subprocess.run(["cmd.exe", "/k", activation_script])
    else:
        activation_script = f"{VENV_NAME}/bin/activate"
        print("To activate the environment, run:")
        print(f"source {activation_script}")


def main():
    root_path = os.path.abspath(os.path.dirname(__file__))
    create_virtual_environment()
    install_requirements(root_path)
    generate_meshes_for_experiments(root_path)
    make_c_library(root_path)
    activate_environment()
    sys.exit(0)


if __name__ == "__main__":
    main()
