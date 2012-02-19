/*
 * Copyright 2012, Jim Bosch and the LSST Corporation
 * 
 * ndarray is available under two licenses, both of which are described
 * more fully in other files that should be distributed along with
 * the code:
 * 
 *  - A simple BSD-style license (ndarray-bsd-license.txt); under this
 *    license ndarray is broadly compatible with essentially any other
 *    code.
 * 
 *  - As a part of the LSST data management software system, ndarray is
 *    licensed with under the GPL v3 (LsstLicenseStatement.txt).
 * 
 * These files can also be found in the source distribution at:
 * 
 * https://github.com/ndarray/ndarray
 */
#ifndef NDARRAY_python_h_INCLUDED
#define NDARRAY_python_h_INCLUDED

/**
 *  @file ndarray/python.h
 *  @brief Public header file for ndarray Python support.
 *
 *  \warning Both the main Python C-API header, "Python.h", and the
 *  Numpy C-API headers "arrayobject.h" and "ufuncobject.h" must
 *  be included before ndarray/python.hpp or any of the files in
 *  ndarray/python.
 *
 *  \note This file is not included by the main "ndarray.h" header file.
 */

/** \defgroup ndarrayndarrayPythonGroup Python Support
 *
 *  The ndarray Python support module provides conversion
 *  functions between ndarray objects, notably Array and
 *  Vector, and Python Numpy objects.
 */

/// @internal \defgroup ndarrayPythonInternalGroup Python Support Internals

#include "Python.h"
#include "ndarray.h"
#include "ndarray/python/numpy.h"
#include "ndarray/python/ufunctors.h"
#include "ndarray/python/Vector.h"

#endif // !NDARRAY_python_h_INCLUDED
