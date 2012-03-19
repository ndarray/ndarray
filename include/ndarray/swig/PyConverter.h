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

#ifndef NDARRAY_SWIG_PyConverter_h_INCLUDED
#define NDARRAY_SWIG_PyConverter_h_INCLUDED

/**
 *  @file ndarray/swig/PyConverter.h
 *  @brief Python C-API conversions for standard numeric types.
 */
#include <Python.h>

#include <boost/intrusive_ptr.hpp>
#include <complex>

#ifndef DOXYGEN
//namespace boost {
inline void intrusive_ptr_add_ref(PyObject * obj) { Py_INCREF(obj); }
inline void intrusive_ptr_release(PyObject * obj) { Py_DECREF(obj); }
//}
#endif

namespace ndarray {

/**
 *  @brief A reference-counting smart pointer for PyObject.
 */
typedef boost::intrusive_ptr<PyObject> PyPtr;

template <typename T> struct PyConverter;

namespace detail {

/**
 *  @internal @ingroup ndarrayPythonInternalGroup
 *  @brief Non-specialized base class for PyConverter.
 */
template <typename T>
struct PyConverterBase {
    
    /**
     *  @brief Check if a Python object might be convertible to T.
     *
     *  \return true if the conversion may be successful, and 
     *  false if it definitely is not.  Will not raise a Python
     *  exception.
     *
     *  This is mostly useful for wrapper generators like SWIG or
     *  Boost.Python, which could use matches() to check if an
     *  Python arguments match a particular signature for an
     *  overloaded C++ function.
     *
     *  \sa PyConverter<T>::fromPythonStage1()
     */
    static bool matches(
        PyObject * arg // input Python object (borrowed)
    ) {
        PyPtr p(arg,true);
        if (!PyConverter<T>::fromPythonStage1(p)) {
            PyErr_Clear();
            return false;
        }
        return true;
    }

    /**
     *  @brief Convert a Python object to a C++ object.
     *
     *  fromPython() is appropriate for use with the PyArg_ParseX
     *  Python C-API functions as a "O&" converter function.
     *
     *  \return true if the conversion was successful, and false
     *  otherwise (with a Python exception set).
     *
     *  \sa PyConverter<T>::fromPythonStage1()
     *  \sa PyConverter<T>::fromPythonStage2()
     **/
    static int fromPython(
        PyObject * arg,  // input Python object (borrowed)
        T * output       // pointer to an existing C++
    ) {
        PyPtr p(arg,true);
        if (!PyConverter<T>::fromPythonStage1(p)) return false;
        return PyConverter<T>::fromPythonStage2(arg,*output);
    }
    
};

} // namespace ndarray::detail

/**
 *  @ingroup ndarrayndarrayPythonGroup
 *  @brief A class providing Python conversion functions for T.
 *
 *  Undocumented specializations exist for bool, int, long, float, double,
 *  std::complex, and std::string.
 */
template <typename T>
struct PyConverter : public detail::PyConverterBase<T> {

    /** 
     *  @brief Convert a C++ object to a new Python object.
     *
     *  \return A new Python object, or NULL on failure (with
     *  a Python exception set).
     */
    static PyObject * toPython(
        T const & input ///< Input C++ object.
    );

    /**
     *  @brief Return the Python TypeObject that corresponds to
     *  the object the toPython() function returns.
     */
    static PyTypeObject const * getPyType();

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
    );

    /**
     *  @brief Complete a Python to C++ conversion begun
     *  with fromPythonStage1().
     *
     *  \return true if the conversion was successful,
     *  and false otherwise (with a Python exception set).
     */
    static bool fromPythonStage2(
        PyPtr const & p, ///< A Python object processed by fromPythonStage1().
        T & output       ///< The output C++ object.
    );

};

/// \cond SPECIALIZATIONS

template <>
struct PyConverter<bool> : public detail::PyConverterBase<bool> {

    static bool fromPythonStage1(PyPtr & input) {
        if (!PyBool_Check(input.get())) {
            PyPtr s(PyObject_Repr(input.get()));
            if (!s) return false;
            char * cs = PyString_AsString(s.get());
            if (!cs) return false;
            PyErr_Format(PyExc_TypeError,"'%s' is not a valid C++ bool value.",cs);
        }
        return true;
    }

    static bool fromPythonStage2(PyPtr const & input, bool & output) {
        NDARRAY_ASSERT(input);
        output = (input.get() == Py_True);
        return true;
    }

    static PyObject * toPython(bool input) {
        if (input) Py_RETURN_TRUE;
        Py_RETURN_FALSE;
    }

    static PyTypeObject const * getPyType() { return &PyBool_Type; }
};

template <>
struct PyConverter<int> : public detail::PyConverterBase<int> {

    static bool fromPythonStage1(PyPtr & input) {
        if (!PyInt_Check(input.get()) && !PyLong_Check(input.get())) {
            PyPtr s(PyObject_Repr(input.get()));
            if (!s) return false;
            char * cs = PyString_AsString(s.get());
            if (!cs) return false;
            PyErr_Format(PyExc_TypeError,"'%s' is not a valid C++ int value.",cs);
        }
        return true;
    }

    static bool fromPythonStage2(PyPtr const & input, int & output) {
        NDARRAY_ASSERT(input);
        output = PyInt_AsLong(input.get());
        return true;
    }

    static PyObject * toPython(int input) {
        return PyInt_FromLong(input);
    }
    
    static PyTypeObject const * getPyType() { return &PyInt_Type; }
};

template <>
struct PyConverter<long> : public detail::PyConverterBase<long> {

    static bool fromPythonStage1(PyPtr & input) {
        if (!PyInt_Check(input.get()) && !PyLong_Check(input.get())) {
            PyPtr s(PyObject_Repr(input.get()));
            if (!s) return false;
            char * cs = PyString_AsString(s.get());
            if (!cs) return false;
            PyErr_Format(PyExc_TypeError,"'%s' is not a valid C++ long value.",cs);
        }
        return true;
    }

    static bool fromPythonStage2(PyPtr const & input, long & output) {
        NDARRAY_ASSERT(input);
        output = PyLong_AsLong(input.get());
        return true;
    }

    static PyObject * toPython(long input) {
        return PyLong_FromLong(input);
    }

    static PyTypeObject const * getPyType() { return &PyLong_Type; }
};

template <>
struct PyConverter<float> : public detail::PyConverterBase<float> {

    static bool fromPythonStage1(PyPtr & input) {
        if (!PyFloat_Check(input.get())) {
            PyPtr s(PyObject_Repr(input.get()));
            if (!s) return false;
            char * cs = PyString_AsString(s.get());
            if (!cs) return false;
            PyErr_Format(PyExc_TypeError,"'%s' is not a valid C++ float value.",cs);
        }
        return true;
    }

    static bool fromPythonStage2(PyPtr const & input, float & output) {
        NDARRAY_ASSERT(input);
        output = PyFloat_AsDouble(input.get());
        return true;
    }

    static PyObject * toPython(float input) {
        return PyFloat_FromDouble(input);
    }
    
    static PyTypeObject const * getPyType() { return &PyFloat_Type; }
};

template <>
struct PyConverter<double> : public detail::PyConverterBase<double> {

    static bool fromPythonStage1(PyPtr & input) {
        if (!PyFloat_Check(input.get())) {
            PyPtr s(PyObject_Repr(input.get()));
            if (!s) return false;
            char * cs = PyString_AsString(s.get());
            if (!cs) return false;
            PyErr_Format(PyExc_TypeError,"'%s' is not a valid C++ double value.",cs);
        }
        return true;
    }

    static bool fromPythonStage2(PyPtr const & input, double & output) {
        NDARRAY_ASSERT(input);
        output = PyFloat_AsDouble(input.get());
        return true;
    }

    static PyObject * toPython(double input) {
        return PyFloat_FromDouble(input);
    }
    
    static PyTypeObject const * getPyType() { return &PyFloat_Type; }
};

template <typename U>
struct PyConverter< std::complex<U> > : public detail::PyConverterBase< std::complex<U> > {

    static bool fromPythonStage1(PyPtr & input) {
        if (!PyComplex_Check(input.get())) {
            PyPtr s(PyObject_Repr(input.get()));
            if (!s) return false;
            char * cs = PyString_AsString(s.get());
            if (!cs) return false;
            PyErr_Format(PyExc_TypeError,"'%s' is not a valid C++ complex value.",cs);
        }
        return true;
    }

    static bool fromPythonStage2(PyPtr const & input, std::complex<U> & output) {
        NDARRAY_ASSERT(input);
        output.real() = PyComplex_RealAsDouble(input.get());
        output.imag() = PyComplex_ImagAsDouble(input.get());
        return true;
    }

    static PyObject * toPython(std::complex<U> const & input) {
        return PyComplex_FromDoubles(input.real(),input.imag());
    }
    
    static PyTypeObject const * getPyType() { return &PyComplex_Type; }
};

template <>
struct PyConverter< std::string > : public detail::PyConverterBase<std::string> {

    static bool fromPythonStage1(PyPtr & input) {
        if (!PyString_Check(input.get())) {
            PyPtr s(PyObject_Repr(input.get()));
            if (!s) return false;
            char * cs = PyString_AsString(s.get());
            if (!cs) return false;
            PyErr_Format(PyExc_TypeError,"'%s' is not a valid C++ string value.",cs);
        }
        return true;
    }

    static bool fromPythonStage2(PyPtr const & input, std::string & output) {
        NDARRAY_ASSERT(input);
        char * buf = 0;
        Py_ssize_t size = 0;
        if (PyString_AsStringAndSize(input.get(),&buf,&size) == -1) return false;
        output = std::string(buf,size);
        return true;
    }

    static PyObject * toPython(std::string const & input) {
        return PyString_FromStringAndSize(input.data(),input.size());
    }
    
    static PyTypeObject const * getPyType() { return &PyString_Type; }
};

/// \endcond

} // namespace ndarray

#endif // !NDARRAY_SWIG_PyConverter_h_INCLUDED
