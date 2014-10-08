// -*- c++ -*-
/*
 * Copyright (c) 2010-2012, Jim Bosch
 * All rights reserved.
 *
 * ndarray is distributed under a simple BSD-like license;
 * see the LICENSE file that should be present in the root
 * of the source distribution, or alternately available at:
 * https://github.com/ndarray/ndarray
 */
#include <cstddef>
#include "Python.h"
#include "numpy/arrayobject.h"
#include "ndarray/swig.h"
#ifndef _MSC_VER
#pragma GCC diagnostic ignored "-Wuninitialized"
#endif

template <typename T, int N>
static PyObject * passVector(PyObject * self, PyObject * args) {
    ndarray::Vector<T,N> vector;
    if (!PyArg_ParseTuple(args,"O&",ndarray::PyConverter< ndarray::Vector<T,N> >::fromPython,&vector))
        return NULL;
    return ndarray::PyConverter< ndarray::Vector<T,N> >::toPython(vector);
}

template <typename T, int N, int C>
static PyObject * passArray(PyObject * self, PyObject * args) {
    ndarray::Array<T,N,C> array;
    if (!PyArg_ParseTuple(args,"O&",ndarray::PyConverter< ndarray::Array<T,N,C> >::fromPython,&array))
        return NULL;
    return ndarray::PyConverter< ndarray::Array<T,N,C> >::toPython(array);
}

template <typename T, int N>
static PyObject * makeArray(PyObject * self, PyObject * args) {
    ndarray::Vector<std::size_t,N> shape;
    if (!PyArg_ParseTuple(args,"O&",ndarray::PyConverter< ndarray::Vector<std::size_t,N> >::fromPython,&shape))
        return NULL;
    ndarray::Array<T,N,N> array = ndarray::allocate(shape);
    array.deep() = static_cast<T>(0);
    return ndarray::PyConverter< ndarray::Array<T,N,N> >::toPython(array);
}

static PyMethodDef methods[] = {
    {"passFloatVector3",&passVector<double,3>,METH_VARARGS,NULL},
    {"passFloatArray33",&passArray<double,3,3>,METH_VARARGS,NULL},
    {"passConstFloatArray33",&passArray<double const,3,3>,METH_VARARGS,NULL},
    {"passFloatArray30",&passArray<double,3,0>,METH_VARARGS,NULL},
    {"makeFloatArray3",&makeArray<double,3>,METH_VARARGS,NULL},
    {"passIntVector3",&passVector<int,3>,METH_VARARGS,NULL},
    {"passIntArray33",&passArray<int,3,3>,METH_VARARGS,NULL},
    {"passuIntArray33",&passArray<unsigned int,3,3>,METH_VARARGS,NULL},
    {"passConstIntArray33",&passArray<int const,3,3>,METH_VARARGS,NULL},
    {"passIntArray30",&passArray<int,3,0>,METH_VARARGS,NULL},
    {"makeIntArray3",&makeArray<int,3>,METH_VARARGS,NULL},
    {NULL}
};

extern "C"
PyMODINIT_FUNC
initpython_test_mod(void) {
    import_array();
    PyObject * module = Py_InitModule("python_test_mod",methods);
    if (module == NULL) return;
}
