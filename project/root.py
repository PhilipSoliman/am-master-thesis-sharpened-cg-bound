from os.path import abspath, dirname
from pathlib import Path


def get_root() -> Path:
    file_abs_path = abspath(dirname(__file__))
    return Path(file_abs_path).parent
