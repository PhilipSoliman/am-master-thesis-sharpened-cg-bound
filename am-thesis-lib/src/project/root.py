from os.path import abspath, dirname
from pathlib import Path


def get_project_root() -> Path:
    file_abs_path = abspath(dirname(__file__))
    return Path(file_abs_path)


def get_package_src() -> Path:
    return get_project_root().parent


def get_package_root() -> Path:
    return get_package_src().parent


def get_root() -> Path:
    return get_package_root().parent
