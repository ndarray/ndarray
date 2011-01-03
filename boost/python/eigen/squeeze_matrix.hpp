#ifndef BOOST_PYTHON_EIGEN_SQUEEZE_MATRIX_HPP_INCLUDED
#define BOOST_PYTHON_EIGEN_SQUEEZE_MATRIX_HPP_INCLUDED

#include <boost/python/numpy/numpy.hpp>
#include <Eigen/Core>

namespace boost { namespace python {

template <typename Base = default_call_policies>
struct squeeze_matrix : Base {
    static PyObject * postcall(PyObject *, PyObject * result) {
        object r = object(handle<>(result));
        numpy::ndarray a(extract<numpy::ndarray>(r.attr("base")));
        if (a.get_nd() != 2) {
            PyErr_SetString(PyExc_ValueError, "squeeze_matrix requires a numpy.matrix return value");
            throw_error_already_set();
        }
        if (a.shape(0) == 1) {
            a = a.reshape(make_tuple(a.shape(1)));
        } else if (a.shape(1) == 1) {
            a = a.reshape(make_tuple(a.shape(0)));
        }
        Py_INCREF(a.ptr());
        return a.ptr();
    }
};

}} // namespace boost::python

#endif // !BOOST_PYTHON_EIGEN_SQUEEZE_MATRIX_HPP_INCLUDED
