from setuptools import find_packages, setup, Extension
import numpy

setup(
    name="project",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=[
        Extension(
            "project.solvers.clib.custom_cg",
            ["src/project/solvers/clib/custom_cg.cpp"],
            include_dirs=[numpy.get_include()],
            language="c++"
        )
    ],
)