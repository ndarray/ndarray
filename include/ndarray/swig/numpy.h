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

#ifndef NDARRAY_SWIG_numpy_h_INCLUDED
#define NDARRAY_SWIG_numpy_h_INCLUDED

/** 
 *  @file ndarray/swig/numpy.h
 *  @brief Python C-API conversions between ndarray and numpy.
 */

#include "ndarray.h"
#include "ndarray/swig/PyConverter.h"

namespace ndarray {
namespace detail {

/** 
 *  @internal @ingroup ndarrayPythonInternalGroup
 *  @brief Traits class that specifies Numpy typecodes for numeric types. 
 */
template <typename T> struct NumpyTraits { 
    static int getCode(); 
};

/// \cond SPECIALIZATIONS
    
template <> struct NumpyTraits<bool> {
    static int getCode() {
	if (sizeof(bool)==sizeof(npy_bool)) return NPY_BOOL;
	if (sizeof(bool)==1) return NPY_UBYTE;
	if (sizeof(bool)==2 && sizeof(short)==2) return NPY_USHORT;
	if (sizeof(bool)==4 && sizeof(int)==4) return NPY_UINT;
	assert(false);
	return 0;
    }
};

template <> struct NumpyTraits<npy_ubyte> { static int getCode() { return NPY_UBYTE; } };
template <> struct NumpyTraits<npy_byte> { static int getCode() { return NPY_BYTE; } };
template <> struct NumpyTraits<npy_ushort> { static int getCode() { return NPY_USHORT; } };
template <> struct NumpyTraits<npy_short> { static int getCode() { return NPY_SHORT; } };
template <> struct NumpyTraits<npy_uint> { static int getCode() { return NPY_UINT; } };
template <> struct NumpyTraits<npy_int> { static int getCode() { return NPY_INT; } };
template <> struct NumpyTraits<npy_ulong> { static int getCode() { return NPY_ULONG; } };
template <> struct NumpyTraits<npy_long> { static int getCode() { return NPY_LONG; } };
template <> struct NumpyTraits<npy_ulonglong> { static int getCode() { return NPY_ULONGLONG; } };
template <> struct NumpyTraits<npy_longlong> { static int getCode() { return NPY_LONGLONG; } };
template <> struct NumpyTraits<npy_float> { static int getCode() { return NPY_FLOAT; } };
template <> struct NumpyTraits<npy_double> { static int getCode() { return NPY_DOUBLE; } };
template <> struct NumpyTraits<npy_longdouble> { static int getCode() { return NPY_LONGDOUBLE; } };
template <> struct NumpyTraits<npy_cfloat> { static int getCode() { return NPY_CFLOAT; } };
template <> struct NumpyTraits<npy_cdouble> { static int getCode() { return NPY_CDOUBLE; } };
template <> struct NumpyTraits<npy_clongdouble> { static int getCode() { return NPY_CLONGDOUBLE; } };

template <> struct NumpyTraits<std::complex<float> > { 
    static int getCode() { assert(sizeof(std::complex<float>)==sizeof(npy_cfloat)); return NPY_CFLOAT; } 
};

template <> struct NumpyTraits<std::complex<double> > { 
    static int getCode() { assert(sizeof(std::complex<double>)==sizeof(npy_cdouble)); return NPY_CDOUBLE; } 
};

template <> struct NumpyTraits<std::complex<long double> > { 
    static int getCode() { 
	assert(sizeof(std::complex<long double>)==sizeof(npy_clongdouble)); 
	return NPY_CLONGDOUBLE; 
    }
};

/// \endcond

/** 
 *  @internal @ingroup ndarrayPythonInternalGroup
 *  @brief A destructor for a Python CObject that owns a shared_ptr.
 */
inline void destroyCObject(void * p) {
    ndarray::Manager::Ptr * b = reinterpret_cast<ndarray::Manager::Ptr*>(p);
    delete b;
}

} // namespace ndarray::detail

/**
 *  @ingroup ndarrayPythonGroup
 *  @brief A traits class providing Python conversion functions for Array.
 *
 *  This specialization, for Array, adds addititional optional arguments
 *  to the toPython() conversion member function.
 */
template <typename T, int N, int C>
struct PyConverter< Array<T,N,C> > : public detail::PyConverterBase< Array<T,N,C> > {
    typedef typename Array<T,N,C>::Element Element;
    typedef typename boost::remove_const<Element>::type NonConst;

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
        if (!PyArray_Check(p.get())) {
            PyErr_SetString(PyExc_TypeError, "numpy.ndarray argument required");
            return false;
        }
        int actualType = PyArray_TYPE(p.get());
        int requiredType = detail::NumpyTraits<NonConst>::getCode();
        if (actualType != requiredType) {
            PyErr_SetString(PyExc_ValueError, "numpy.ndarray argument has incorrect data type");
            return false;
        }
        if (PyArray_NDIM(p.get()) != N) {
            PyErr_SetString(PyExc_ValueError, "numpy.ndarray argument has incorrect number of dimensions");
            return false;
        }
        bool writeable = !boost::is_const<Element>::value;
        if (writeable && !(PyArray_FLAGS(p.get()) & NPY_WRITEABLE)) {
            PyErr_SetString(PyExc_TypeError, "numpy.ndarray argument must be writeable");
            return false;
        }
        if (C > 0) {
            int requiredStride = sizeof(Element);
            for (int i = 0; i < C; ++i) {
                int actualStride = PyArray_STRIDE(p.get(), N-i-1);
                if (actualStride != requiredStride) {
                    PyErr_SetString(
                        PyExc_ValueError,
                        "numpy.ndarray does not have enough row-major contiguous dimensions"
                    );
                    return false;
                }
                requiredStride *= PyArray_DIM(p.get(), N-i-1);
            }
        } else if (C < 0) {
            int requiredStride = sizeof(Element);
            for (int i = 0; i < -C; ++i) {
                int actualStride = PyArray_STRIDE(p.get(), i);
                if (actualStride != requiredStride) {
                    PyErr_SetString(
                        PyExc_ValueError,
                        "numpy.ndarray does not have enough column-major contiguous dimensions"
                    );
                    return false;
                }
                requiredStride *= PyArray_DIM(p.get(), i);
            }
        }
        return true;
    }

    /**
     *  @brief Complete a Python to C++ conversion begun with fromPythonStage1().
     * 
     *  The copy will be shallow if possible and deep if necessary to meet the data type
     *  and contiguousness requirements.  If a non-const array is required, the copy will
     *  always be shallow; if this is not possible, ValueError will be raised.
     *
     *  The output Array's shared_ptr owner attribute will own a reference to the numpy
     *  array that ultimately owns the data (either the original or the copy).
     *
     *  \return true on success, false on failure (with a Python exception set).
     */
    static bool fromPythonStage2(
        PyPtr const & input,  ///< Result of fromPythonStage1().
        Array<T,N,C> & output ///< Reference to existing output C++ object.
    ) {
        if (!(PyArray_FLAGS(input.get()) & NPY_ALIGNED)) {
            PyErr_SetString(PyExc_ValueError, "unaligned arrays cannot be converted to C++");
            return false;
        }
        Vector<int,N> shape;
        Vector<int,N> strides;
        std::copy(PyArray_DIMS(input.get()), PyArray_DIMS(input.get()) + N, shape.begin());
        std::copy(PyArray_STRIDES(input.get()), PyArray_STRIDES(input.get()) + N , strides.begin());
        for (int i = 0; i < N; ++i) strides[i] /= sizeof(Element);
        output = external(
            reinterpret_cast<Element*>(PyArray_DATA(input.get())),
            shape, strides, input
        );
        return true;
    }

    /**
     *  @brief Create a numpy.ndarray from an ndarray::Array.
     *
     *  The Array will be shallow-copied with reference counting if either
     *  m.getManager() is not empty or the optional owner argument is supplied;
     *  otherwise a deep copy will be made.
     *
     *  \return a new Python object, or NULL on failure (with
     *  a Python exception set).
     */
    static PyObject* toPython(
        Array<T,N,C> const & m, ///< The input Array object.
        PyObject* owner=NULL    /**< A Python object that owns the memory in the Array.
                                 *   If NULL, one will be constructed from m.getManager(). */
    ) {
        int flags = NPY_ALIGNED;
        if (C==N) flags |= NPY_C_CONTIGUOUS;
        bool writeable = !boost::is_const<Element>::value;
        if (writeable) flags |= NPY_WRITEABLE;
        npy_intp outShape[N];
        npy_intp outStrides[N];
        Vector<int,N> inShape = m.getShape();
        Vector<int,N> inStrides = m.getStrides();
        std::copy(inShape.begin(), inShape.end(), outShape);
        for (int i = 0; i < N; ++i) outStrides[i] = inStrides[i] * sizeof(Element);
        PyPtr array(
            PyArray_New(
                &PyArray_Type, N, outShape, detail::NumpyTraits<NonConst>::getCode(),
                outStrides, const_cast<NonConst*>(m.getData()), sizeof(Element), flags, NULL
            ),
            false
        );
        if (!array) return NULL;
        if (!m.getManager() && owner == NULL) {
            flags = NPY_CARRAY_RO | NPY_ENSURECOPY | NPY_C_CONTIGUOUS;
            if (writeable) flags |= NPY_WRITEABLE;
            PyPtr r = PyArray_FROM_OF(array.get(),flags);
            if (!r) return NULL;
            array.swap(r);
        } else {
            if (owner != NULL) {
                Py_INCREF(owner);
            } else {
                owner = PyCObject_FromVoidPtr(new Manager::Ptr(m.getManager()), detail::destroyCObject);
            }
            reinterpret_cast<PyArrayObject*>(array.get())->base = owner;
        }
        Py_INCREF(array.get());
        return PyArray_Return(reinterpret_cast<PyArrayObject*>(array.get()));
    }

    static PyTypeObject const * getPyType() { return &PyArray_Type; }
};

} // namespace ndarray

#endif // !NDARRAY_SWIG_numpy_h_INCLUDED
