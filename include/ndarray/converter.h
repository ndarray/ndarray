/* 
 * LSST Data Management System
 * Copyright 2008, 2009, 2010 LSST Corporation.
 * 
 * This product includes software developed by the
 * LSST Project (http://www.lsst.org/).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the LSST License Statement and 
 * the GNU General Public License along with this program.  If not, 
 * see <http://www.lsstcorp.org/LegalNotices/>.
 */

#ifndef NDARRAY_converter_h_INCLUDED
#define NDARRAY_converter_h_INCLUDED

/**
 *  @file ndarray/converter.h
 *  @brief Public header file for SWIG-based Python support.
 *
 *  \warning Both the Numpy C-API headers "arrayobject.h" and
 *  "ufuncobject.h" must be included before ndarray/python.hpp
 *  or any of the files in ndarray/python.
 *
 *  \note This file is not included by the main "ndarray.h" header file.
 */

/** \defgroup ndarrayPythonGroup Python Support
 *
 *  The ndarray Python support module provides conversion
 *  functions between ndarray objects, notably Array and
 *  Vector, and Python Numpy objects.
 */

#include "Python.h"
#include "ndarray.h"
#include "ndarray/converter/numpy.h"
#include "ndarray/converter/ufunctors.h"
#include "ndarray/converter/Vector.h"

#endif // !NDARRAY_converter_h_INCLUDED
