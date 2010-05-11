#ifndef NDARRAY_python_hpp_INCLUDED
#define NDARRAY_python_hpp_INCLUDED

/**
 *  @file ndarray/python.hpp
 *  @brief Public header file for ndarray Python support.
 *
 *  \warning Both the main Python C-API header, "Python.h", and the
 *  Numpy C-API headers "arrayobject.h" and "ufuncobject.h" must
 *  be included before ndarray/python.hpp or any of the files in
 *  ndarray/python.
 *
 *  \note This file is not included by the main "ndarray.hpp" header file.
 */

/** \defgroup PythonGroup Python Support
 *
 *  The ndarray Python support module provides conversion
 *  functions between ndarray objects, notably Array and
 *  Vector, and Python Numpy objects.
 *
 *  \note initializePython() must be called before using
 *  the rest of the ndarray Python support library.
 */

/// @internal \defgroup PythonInternalGroup Python Support Internals

#include "Python.h"
#include "ndarray.hpp"
#include "ndarray/python/numpy.hpp"
#include "ndarray/python/ufunctors.hpp"
#include "ndarray/python/Vector.hpp"

namespace ndarray {

/**
 *  @brief Initialize ndarray Python support.
 *
 *  This function must be called in the module initialization
 *  section of any C++ extension module that uses ndarray,
 *  before any other parts of the ndarray Python support
 *  library are used.
 *
 *  @ingroup PythonGroup
 */
void initializePython();

}

#endif // !NDARRAY_python_hpp_INCLUDED
