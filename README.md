ndarray: NumPy-friendly multidimensional arrays in C++
======================================================
[![Build Status](https://travis-ci.org/ndarray/ndarray.svg?branch=master)](https://travis-ci.org/ndarray/ndarray)

ndarray is a template library that provides multidimensional array
objects in C++, with an interface and features designed to mimic the
Python 'numpy' package as much as possible.

More information can be found in the [documentation at
ndarray.github.io/ndarray](http://ndarray.github.io/ndarray/).


Building from Git
-----------------

CMake is the supported way to build ndarray.

To build with cmake, do:

    mkdir build
    cd build
    cmake ..
    make
    make test

Inclusion and testing of optional dependencies is controlled by NDARRAY_* cmake
options. Dependency resolution can be controlled by the PYBIND11_DIR,
EIGEN_DIR, and FFTW_DIR environment variables. For example, to build with an
alternate Eigen3 install location and disable FFTW testing replace `cmake ..`
with `EIGEN_DIR=/opt/local cmake -DNDARRY_FFTW=OFF ..`.

ndarray::EigenView is not compatible with Eigen >= 3.3,
nor is it compatible with pybind11's default Eigen wrappers.
To build without ndarray::EigenView use cmake option `-DNDARRAY_EIGENVIEW=OFF`

ndarray's build system does not produce the correct suffixes for pybind11 outputs under pybind11 2.1.x (due to a bug in pybind11 itself).  To avoid this problem, please upgrade to pybind11 2.2.x, or try the (now reverted) patch from ndarray commit f46c0f0ff876ceab5aaa3286e5f6e86902e72feb.

However, the cmake build does not support Boost.NumPy library
and Boost.Python wrappers, which are deprecated in favor of pybind11.

Version 1.4.2 of ndarray is the last version to support SWIG.


Building from Compressed Source
-------------------------------

GitHub's automatically generated tarballs and zip files don't include
the Boost.NumPy submodule or the git metadata needed to run "git
submodule", so these features can't be used from release tarball.  Please download from git if you want to use the Boost.NumPy interface.
