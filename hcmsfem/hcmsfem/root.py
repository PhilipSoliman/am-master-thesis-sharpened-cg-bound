from os.path import abspath, dirname
from pathlib import Path


def get_package_dir() -> Path:
    file_abs_path = abspath(dirname(__file__))
    return Path(file_abs_path)


def get_package_root() -> Path:
    return get_package_dir().parent


def get_root() -> Path:
    return get_package_root().parent
