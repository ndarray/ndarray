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
#ifndef BOOST_PYTHON_EIGEN_PYTHONMATRIX_HPP_INCLUDED
#define BOOST_PYTHON_EIGEN_PYTHONMATRIX_HPP_INCLUDED

#include <boost/python/numpy.hpp>
#include <Eigen/Core>

namespace Eigen {

template <typename _Scalar, int _Rows, int _Cols> class PythonMatrix;

template <typename _Scalar, int _Rows, int _Cols> 
struct ei_traits< PythonMatrix<_Scalar,_Rows,_Cols> > {
    typedef _Scalar Scalar;
    enum {
        RowsAtCompileTime = _Rows,
        ColsAtCompileTime = _Cols,
        MaxRowsAtCompileTime = _Rows,
        MaxColsAtCompileTime = _Cols,
        Flags = 0,
        CoeffReadCost = NumTraits<Scalar>::ReadCost
    };
};

template <typename _Scalar, int _Rows, int _Cols>
class PythonMatrix : public MatrixBase< PythonMatrix<_Scalar,_Rows,_Cols> > {
public:
    
    EIGEN_GENERIC_PUBLIC_INTERFACE(PythonMatrix);

    explicit PythonMatrix(boost::python::numpy::matrix const & py) : _py(py) {
        if ((_Rows != Eigen::Dynamic && _Rows != _py.shape(0))
            || (_Cols != Eigen::Dynamic && _Cols != _py.shape(1))) {
            if ((_Rows == 1 && _py.shape(1) == 1 && (_Cols == Eigen::Dynamic || _Cols == _py.shape(0))) 
                || (_Cols == 1 && _py.shape(0) == 1 && (_Rows == Eigen::Dynamic || _Rows == _py.shape(1)))) {
                _py = py.transpose();
            } else {
                PyErr_SetString(PyExc_ValueError, "Incorrect shape for fixed-size matrix.");
                boost::python::throw_error_already_set();
            }
        }
        if (_py.get_dtype() != boost::python::numpy::dtype::get_builtin<_Scalar>()) {
            PyErr_SetString(PyExc_TypeError, "Incorrect data type for matrix.");
            boost::python::throw_error_already_set();
        }
    }

    EIGEN_INHERIT_ASSIGNMENT_OPERATORS(PythonMatrix);

    Scalar * data() { return _py.get_data(); }
    Scalar const * data() const { return _py.get_data(); }

    inline int rows() const { return _py.shape(0); }
    inline int cols() const { return _py.shape(1); }

    inline Scalar & coeffRef(int row, int col) {
        return *reinterpret_cast<Scalar*>(_py.get_data() + row * _py.strides(0) + col * _py.strides(1));
    }

    inline Scalar const coeff(int row, int col) const {
        return *reinterpret_cast<Scalar*>(_py.get_data() + row * _py.strides(0) + col * _py.strides(1));
    }

    inline Scalar & coeffRef(int index) {
        if (_Rows == 1)
            return *reinterpret_cast<Scalar*>(_py.get_data() + index * _py.strides(1));
        else if (_Cols == 1)
            return *reinterpret_cast<Scalar*>(_py.get_data() + index * _py.strides(0));
        PyErr_SetString(PyExc_TypeError, "Cannot index PythonMatrix with a single integer.");
        boost::python::throw_error_already_set();
    }

    inline Scalar const coeff(int index) const {
        if (_Rows == 1)
            return *reinterpret_cast<Scalar*>(_py.get_data() + index * _py.strides(1));
        else if (_Cols == 1)
            return *reinterpret_cast<Scalar*>(_py.get_data() + index * _py.strides(0));
        PyErr_SetString(PyExc_TypeError, "Cannot index PythonMatrix with a single integer.");
        boost::python::throw_error_already_set();
    }

    boost::python::numpy::matrix getPyObject() const { return _py; }

private:
    boost::python::numpy::matrix _py;
};

} // namespace Eigen

namespace boost { namespace python {

template <typename _Scalar, int _Rows, int _Cols>
struct to_python_value< Eigen::PythonMatrix<_Scalar,_Rows,_Cols> const & > 
    : public detail::builtin_to_python
{
    inline PyObject * operator()(Eigen::PythonMatrix<_Scalar,_Rows,_Cols> const & x) const {
        numpy::matrix obj(x.getPyObject());
        Py_INCREF(obj.ptr());
        return obj.ptr();
    }
    inline PyTypeObject const * get_pytype() const {
        return converter::object_manager_traits<numpy::matrix>::get_pytype();
    }
};

template <typename _Scalar, int _Rows, int _Cols>
struct to_python_value< Eigen::PythonMatrix<_Scalar,_Rows,_Cols> & > 
    : public detail::builtin_to_python
{
    inline PyObject * operator()(Eigen::PythonMatrix<_Scalar,_Rows,_Cols> & x) const {
        numpy::matrix obj(x.getPyObject());
        Py_INCREF(obj.ptr());
        return obj.ptr();
    }
    inline PyTypeObject const * get_pytype() const {
        return converter::object_manager_traits<numpy::matrix>::get_pytype();
    }
};

namespace converter {

template <typename _Scalar, int _Rows, int _Cols>
struct arg_to_python< Eigen::PythonMatrix<_Scalar,_Rows,_Cols> > : public handle<> {
    inline arg_to_python(Eigen::PythonMatrix<_Scalar,_Rows,_Cols> const & v) :
        handle<>(borrowed(v.getPyObject().ptr())) {}
};

template <typename _Scalar, int _Rows, int _Cols>
struct arg_rvalue_from_python< Eigen::PythonMatrix<_Scalar,_Rows,_Cols> const & > {
    typedef Eigen::PythonMatrix<_Scalar,_Rows,_Cols> result_type;

    arg_rvalue_from_python(PyObject * p) : _p(borrowed(p)) {}

    bool convertible() const { return true; }

    result_type operator()() const {
        boost::python::numpy::matrix m(python::detail::borrowed_reference(_p.get()));
        return Eigen::PythonMatrix<_Scalar,_Rows,_Cols>(m);
    }

private:
    mutable handle<> _p;
};

} // namespace boost::python::converter

}} // namespace boost::python

#endif // !BOOST_PYTHON_EIGEN_PYTHONMATRIX_HPP_INCLUDED
