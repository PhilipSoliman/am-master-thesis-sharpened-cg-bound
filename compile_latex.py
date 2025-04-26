import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.resolve()
VENV_PATH = REPO_ROOT / ".venv"
FIGURES_FOLDER = REPO_ROOT / "figures"
GENERATE_FIGURES_SCRIPT = REPO_ROOT / "generate_figures.py"
SETUP_ENV_SCRIPT = REPO_ROOT / "setup_env.py"


def run_command(
    command, cwd=None, log_file: Path | None = None, env=None, stream=False
):
    if log_file and not stream:
        with open(log_file, "a") as log:
            result = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
    elif log_file is None and stream:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            env=env,
            capture_output=False,
        )
    else:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            env=env,
            capture_output=True,
        )

    if result.returncode != 0:
        print(f"Command failed: {command}")
        if log_file:
            print(f"See log: {log_file}")
        sys.exit(1)


def ensure_figures_folder():
    if FIGURES_FOLDER.exists():
        return

    if not VENV_PATH.exists():
        print(
            f"Virtual environment {VENV_PATH} not found.\n\nPlease run {SETUP_ENV_SCRIPT} using python first.\n\nFor example 'py {SETUP_ENV_SCRIPT}'. Then, rerun this script."
        )
        sys.exit(1)

    # Always re-define after setup
    if os.name == "nt":
        venv_python = VENV_PATH / "Scripts" / "python.exe"
        venv_bin = VENV_PATH / "Scripts"
    else:
        venv_python = VENV_PATH / "bin" / "python"
        venv_bin = VENV_PATH / "bin"

    # Set environment PATH correctly
    env = os.environ.copy()
    env["PATH"] = str(venv_bin) + os.pathsep + env["PATH"]

    print("Generating figures...")
    run_command(
        f'"{venv_python}" "{GENERATE_FIGURES_SCRIPT}"',
        cwd=REPO_ROOT,
        env=env,
        stream=True,
    )


def find_tex_files():
    tex_files = []
    for folder in REPO_ROOT.iterdir():
        if folder.is_dir():
            tex_file = folder / (folder.name + ".tex")
            if tex_file.exists():
                tex_files.append(tex_file)
    return tex_files


def compile_tex_file(tex_file):
    tex_dir = tex_file.parent
    tex_name = tex_file.stem

    build_dir = tex_dir / "build"
    build_dir.mkdir(exist_ok=True)

    log_file = tex_dir / f"{tex_name}_build.log"
    if log_file.exists():
        log_file.unlink()

    cmds = [
        f"latexmk -synctex=1 -cd -interaction=nonstopmode -file-line-error -lualatex -outdir={build_dir} {tex_file}",
        f"biber --input-directory={build_dir} --output-directory={build_dir} {tex_name}",
        f"latexmk -synctex=1 -cd -interaction=nonstopmode -file-line-error -lualatex -outdir={build_dir} {tex_file}",
        f"latexmk -synctex=1 -cd -interaction=nonstopmode -file-line-error -lualatex -outdir={build_dir} {tex_file}",
    ]

    wds = [REPO_ROOT, REPO_ROOT / tex_dir, REPO_ROOT, REPO_ROOT]

    for cmd, wd in zip(cmds, wds):
        run_command(cmd, cwd=wd, log_file=log_file)

    # Cleanup
    clean_cmd = f"latexmk -outdir={build_dir} -c {tex_file}"
    run_command(clean_cmd, cwd=REPO_ROOT, log_file=log_file)


def print_progress(current, total, filename) -> int:
    bar_length = 40
    filled_length = int(bar_length * current // total)
    bar = "█" * filled_length + "-" * (bar_length - filled_length)
    line = f"\rCompiling [{bar}] {current}/{total} files | {filename}"
    # print(line, end="", flush=True)
    print(f"\r{line}", end="", flush=True)
    return len(line)


def choose_files(tex_files):
    print("\nAvailable .tex files:")
    for idx, file in enumerate(tex_files, start=1):
        print(f"{idx}: {file.relative_to(REPO_ROOT)}")

    while True:
        choice = input(
            "\nChoose a number to compile a specific file or 'a' to compile all: "
        ).strip()
        if choice.lower() == "a":
            return tex_files
        elif choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(tex_files):
                return [tex_files[idx]]
        print("Invalid input. Try again.")


def main():
    ensure_figures_folder()

    tex_files = find_tex_files()
    if not tex_files:
        print("No matching .tex files found.")
        sys.exit(0)

    selected_files = choose_files(tex_files)

    total = len(selected_files)
    print()
    for idx, tex_file in enumerate(selected_files, start=1):
        line_length = print_progress(idx, total, tex_file.stem)
        compile_tex_file(tex_file)
        # Clear the progress line after completion
        print("\r" + " " * line_length + "\r", end="", flush=True)

    print("All requested .tex files compiled successfully.")


if __name__ == "__main__":
    main()
