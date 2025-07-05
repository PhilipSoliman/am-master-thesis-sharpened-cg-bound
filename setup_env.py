import os
import re
import subprocess
import sys
import venv

VENV_NAME = ".venv"
LOCAL_PACKAGE_FOLDER = "project"
CLIB_RELPATH = os.path.join(LOCAL_PACKAGE_FOLDER, "solvers", "clib")
PYTHON_EXEC = (
    os.path.join(VENV_NAME, "Scripts", "python.exe")
    if os.name == "nt"
    else os.path.join(VENV_NAME, "bin", "python")
)
CUDA_VERSIONS = ["cu118", "cu126", "cu128"]


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
    cuda_version = _get_cuda_version()
    cuda_used = False
    print(f"Detected CUDA version: {cuda_version}")
    if cuda_version in CUDA_VERSIONS:
        cuda_used = True
        torch_url = f"https://download.pytorch.org/whl/{cuda_version}"
    else:
        torch_url = "https://download.pytorch.org/whl/cpu"

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
    print(
        f"Installed PyTorch {'with CUDA support' if cuda_used else 'without CUDA support'}"
    )

    subprocess.check_call([PYTHON_EXEC, "-m", "pip", "install", "-e", "."])
    print("Installed local packages")


def generate_meshes_for_experiments(root_path):
    mesh_script = os.path.join(root_path, "generate_meshes.py")
    user_input = input(f"Generate meshes for experiments? Yes/No [y/n]: ")

    if user_input.lower() != "y":
        print("Skipping mesh generation.")
        return

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
    lib_path = os.path.join(root_path, CLIB_RELPATH)
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
    subprocess.run(
        [PYTHON_EXEC, "activate_env.py"],
        shell=True,
        check=True,
    )


def _get_cuda_version():
    try:
        output = subprocess.check_output(["nvcc", "--version"]).decode()
        for line in output.split("\n"):
            if "release" in line:
                match = re.search(r"release (\d+)\.(\d+)", line)
                if match:
                    major, minor = match.groups()
                    version = f"{major}{minor}"
                    return f"cu{version}"
                else:
                    raise ValueError("Could not parse CUDA version from nvcc output.")
    except FileNotFoundError:
        input(
            "CUDA not found. If you want to speed up some calculations in this project, please install CUDA. Press Enter to continue without CUDA."
        )
        return ""


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
