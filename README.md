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

However, the cmake build does not support Boost.NumPy library
and Boost.Python wrappers, which are deprecated in favor of pybind11.
If you are keen to use Boost.Python you can try the older
scons build files, as follows

ndarray includes the Boost.NumPy library using git's "submodules"
feature.  When you clone the ndarray repository with git, you'll get
an empty Boost.NumPy directory.  Even if you don't plan to use
Boost.NumPy (which is required only if you want to build the
Boost.Python bindings for ndarray), it *is* necessary to checkout the
Boost.NumPy source (as parts of the build system is shared).  So,
immediately after cloning ndarray, you'll need to run:

git submodule update --init --recursive

From there, you'll be able to build ndarray and (optionally)
Boost.NumPy together just by running "scons" from the root of the
ndarray clone.

Version 1.4.2 of ndarray is the last version to support SWIG.


Building from Compressed Source
-------------------------------

GitHub's automatically generated tarballs and zip files don't include
the Boost.NumPy submodule or the git metadata needed to run "git
submodule", so these features can't be used from release tarball.  Please download from git if you want to use the Boost.NumPy interface.
