from distutils.core import setup
from Cython.Build import cythonize

setup(
    name = "bhtsne",
    ext_modules = cythonize('*.pyx')
)
