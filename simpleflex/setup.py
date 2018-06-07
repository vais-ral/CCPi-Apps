#!/usr/bin/env python

import setuptools
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import os
import sys
import numpy
import platform	

cil_version=os.environ['CIL_VERSION']
if  cil_version == '':
    print("Please set the environmental variable CIL_VERSION")
    sys.exit(1)
	
library_include_path = ""
library_lib_path = ""
try:
    library_include_path = os.environ['LIBRARY_INC']
    library_lib_path = os.environ['LIBRARY_LIB']
except:
    library_include_path = os.environ['PREFIX']+'/include'
    pass
    
extra_include_dirs = [library_include_path, numpy.get_include()]
extra_compile_args = []
extra_library_dirs = []
extra_compile_args = []
extra_link_args = []
extra_libraries = []


if platform.system() == 'Windows':				   
    extra_compile_args[0:] = ['/DWIN32','/EHsc' ]   
else:
    extra_compile_args = ['-fopenmp','-O2', '-funsigned-char', '-Wall', '-std=c++0x']
    
    
setup(
    name='ccpi-apps',
	description='CCPi Core Imaging Library - Image Regularizers',
	version=cil_version,
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("ccpi.apps.simpleflex2polydata",
                             sources=[os.path.join("." , "src", "simpleflex2polydata.pyx" ) ],
                             include_dirs=extra_include_dirs, 
							 library_dirs=extra_include_dirs, 
							 extra_compile_args=extra_compile_args, 
							 libraries=extra_libraries ), 
    
    ],
	zip_safe = False,	
	packages = {'ccpi','ccpi.apps', 'ccpi.apps.simpleflex'},
)

