from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension(
        "add_matrices",
        sources=["add_matrices_wrapper.pyx", "matrix.cu"],
        include_dirs=[numpy.get_include()],
        language='c++',
        extra_compile_args=['-O2', '--ptxas-options=-v', '-arch=sm_60', '--compiler-options', "'-fPIC'"],
        extra_link_args=['-lcudart'],
        library_dirs=['/usr/local/cuda/lib64']
    )
]

setup(
    name='Add Matrices',
    ext_modules=cythonize(ext_modules),
)