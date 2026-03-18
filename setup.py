import sys

from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup
import numpy

force = False
profile = False

if "--force" in sys.argv:
    force = True
    del sys.argv[sys.argv.index("--force")]

if "--profile" in sys.argv:
    profile = True
    del sys.argv[sys.argv.index("--profile")]

if profile:
    directives = {"profile": True}
else:
    directives = {}


def cython_extensions():
    include_dirs = [".", numpy.get_include()]
    module_names = [
        "hdg_postprocess.core.mesh.locator",
        "hdg_postprocess.routines._interpolators_fast",
    ]
    return [
        Extension(module_name, [f"{module_name.replace('.', '/')}.pyx"], include_dirs=include_dirs)
        for module_name in module_names
    ]


setup(
    packages=find_packages(),
    ext_modules=cythonize(cython_extensions(), force=force, compiler_directives=directives),
)
