#ifndef LSST_NDARRAY_python_hpp_INCLUDED
#define LSST_NDARRAY_python_hpp_INCLUDED

/**
 *  @file lsst/ndarray/python.hpp
 *  @brief Public header file for ndarray Python support.
 *
 *  \warning Both the main Python C-API header, "Python.h", and the
 *  Numpy C-API headers "arrayobject.h" and "ufuncobject.h" must
 *  be included before ndarray/python.hpp or any of the files in
 *  ndarray/python.
 *
 *  \note This file is not included by the main "lsst/ndarray.hpp" header file.
 */

/** \defgroup PythonGroup Python Support
 *
 *  The ndarray Python support module provides conversion
 *  functions between ndarray objects, notably Array and
 *  Vector, and Python Numpy objects.
 */

/// @internal \defgroup PythonInternalGroup Python Support Internals

#include "Python.h"
#include "lsst/ndarray.hpp"
#include "lsst/ndarray/python/numpy.hpp"
#include "lsst/ndarray/python/ufunctors.hpp"
#include "lsst/ndarray/python/Vector.hpp"

#endif // !LSST_NDARRAY_python_hpp_INCLUDED
