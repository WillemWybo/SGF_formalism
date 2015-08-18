from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy

setup(ext_modules=[Extension("SGFModel", ["SGFModel.pyx", "sgfCode.cc"], 
								language="c++", extra_compile_args=["-w", "-O3"],
								include_dirs=[numpy.get_include()])], cmdclass={'build_ext': build_ext})