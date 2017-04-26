from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy


extensions = [
    Extension(
        'colvars_interface', ['colvars_interface.pyx'],
        libraries=['m'],
        include_dirs=[numpy.get_include()]
    ),
    Extension(
        'colvars', ['colvars.pyx'],
        libraries=['m'],
        include_dirs=[numpy.get_include()]
    ),
    Extension(
        'xyz', ['xyz.pyx'],
    ),
    Extension(
        'zdcvdt', ['zdcvdt.pyx'],
        libraries=['gsl', 'openblas', 'm'],
        library_dirs=['/opt/OpenBLAS/lib'],
        include_dirs=['/opt/OpenBLAS/include', numpy.get_include()]
    ),
    Extension(
        'zdcvdt_interface', ['zdcvdt_interface.pyx'],
        libraries=['gsl', 'openblas', 'm'],
        library_dirs=['/opt/OpenBLAS/lib'],
        include_dirs=['/opt/OpenBLAS/include', numpy.get_include()]
    ),
]


setup(
    name='icf calculation 2d cation-anion water',
    ext_modules=cythonize(extensions)
)
