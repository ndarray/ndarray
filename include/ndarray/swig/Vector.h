// -*- c++ -*-
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

#ifndef NDARRAY_SWIG_Vector_h_INCLUDED
#define NDARRAY_SWIG_Vector_h_INCLUDED

/** 
 *  @file ndarray/swig/Vector.h
 *  @brief Python C-API conversions for Vector.
*/

#include "ndarray/swig/PyConverter.h"

namespace ndarray {

/**
 *  @ingroup ndarrayndarrayPythonGroup
 *  @brief A traits class providing Python conversion functions for Vector.
 */
template <typename T, int N>
struct PyConverter< Vector<T,N> > : public detail::PyConverterBase< Vector<T,N> > {

    /** 
     *  @brief Convert a Vector to a new Python object.
     *
     *  \return A new Python object, or NULL on failure (with
     *  a Python exception set).
     */
    static PyObject * toPython(
        Vector<T,N> const & input ///< Input C++ object.
    ) {
        PyObject * r = PyTuple_New(N);
        for (int i=0; i<N; ++i) {
            PyTuple_SET_ITEM(r,i,PyConverter<T>::toPython(input[i]));
        }
        return r;
    }

    /**
     *  @brief Return the Python TypeObject that corresponds to
     *  the object the toPython() function returns.
     */
    static PyTypeObject const * getPyType() { return &PyTuple_Type; }

    /**
     *  @brief Check if a Python object is convertible to T
     *  and optionally begin the conversion by replacing the
     *  input with an intermediate.
     *
     *  \return true if a conversion may be possible, and
     *  false if it is not (with a Python exception set).
     */
    static bool fromPythonStage1(
        PyPtr & p /**< On input, a Python object to be converted.
                   *   On output, a Python object to be passed to
                   *   fromPythonStage2().
                   */
    ) {
        if (!PySequence_Check(p.get())) {
            PyErr_Format(PyExc_TypeError,"Expected a Python sequence of length %i.",N);
            return false;
        }
        if (PySequence_Size(p.get()) != N) {
            PyErr_Format(PyExc_ValueError,"Incorrect sequence length for Vector<T,%i>", N);
            return false;
        }
        return true;
    }

    /**
     *  @brief Complete a Python to C++ conversion begun
     *  with fromPythonStage1().
     *
     *  \return true if the conversion was successful,
     *  and false otherwise (with a Python exception set).
     */
    static bool fromPythonStage2(
        PyPtr const & p,           ///< A Python object processed by fromPythonStage1().
        Vector<T,N> & output       ///< The output C++ object.
    ) {
        NDARRAY_ASSERT(p);
        NDARRAY_ASSERT(PySequence_Check(p.get()));
        Vector<T,N> tmp;
        for (int n=0; n<N; ++n) {
            PyPtr item(PySequence_ITEM(p.get(),n),false);
            if (!item) return false;
            if (!PyConverter<T>::fromPythonStage1(item)) return false;
            if (!PyConverter<T>::fromPythonStage2(item,tmp[n])) return false;
        }
        output = tmp;
        return true;
    }

};

} // namespace ndarray

#endif // !NDARRAY_SWIG_Vector_h_INCLUDED
