import os
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm

from utils.utils import get_root

PROJECT_ROOT = get_root()

# get the path to the python executable
PYTHON_PATH = sys.executable

# get all figure generating python files in src and its subdirectories
file_list = []
for root, dirs, files in os.walk(PROJECT_ROOT / "code"):
    for file in files:
        if file.endswith("_fig.py"):
            file_list.append(Path(root) / file)

# run all figure generating python files
progress_bar = tqdm(
    file_list,
    total=len(file_list),
    desc="Running figure generating python files",
    unit="file",
)
for file in progress_bar:
    result = subprocess.run(
        [PYTHON_PATH, file, "--generate-output"], check=True, capture_output=True
    )
    progress_bar.set_postfix_str(f"Finished running {file.name}")
