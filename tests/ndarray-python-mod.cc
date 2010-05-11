#include "Python.h"
#include "numpy/arrayobject.h"
#include "lsst/ndarray/python.hpp"

namespace ndarray = lsst::ndarray;

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
    ndarray::Vector<int,N> shape;
    if (!PyArg_ParseTuple(args,"O&",ndarray::PyConverter< ndarray::Vector<int,N> >::fromPython,&shape))
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
    {NULL}
};

extern "C"
PyMODINIT_FUNC
initndarray_python_test(void) {
    import_array();
    PyObject * module = Py_InitModule("ndarray_python_test",methods);
    if (module == NULL) return;
}
