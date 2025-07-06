from setuptools import setup, Extension, find_packages
import numpy

setup(
    packages=find_packages(where="."),
    package_dir={"": "."},
    ext_modules=[
        Extension(
            "hcmsfem.solvers.clib.custom_cg",
            ["hcmsfem/solvers/clib/custom_cg.cpp"],
            include_dirs=[numpy.get_include()],
            language="c++"
        )
    ],
)