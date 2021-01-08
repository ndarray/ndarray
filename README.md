ndarray: NumPy-friendly multidimensional arrays in C++
======================================================
![Build Status](https://github.com/ndarray/ndarray/workflows/build_and_test/badge.svg)

ndarray is a template library that provides multidimensional array
objects in C++, with an interface and features designed to mimic the
Python 'numpy' package as much as possible.

More information can be found in the [documentation at
ndarray.github.io/ndarray](http://ndarray.github.io/ndarray/).


Installation
------------

ndarray can be built and tested with CMake:

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

ndarray's build system does not produce the correct suffixes for pybind11
outputs under pybind11 2.1.x (due to a bug in pybind11 itself).  To avoid this
problem, please upgrade to pybind11 2.2.x, or try the (now reverted) patch from
ndarray commit f46c0f0ff876ceab5aaa3286e5f6e86902e72feb.

Version 1.4.2 of ndarray is the last version to support SWIG.

Version 1.5.3 of ndarray is the last verison to support Boost.Python.
