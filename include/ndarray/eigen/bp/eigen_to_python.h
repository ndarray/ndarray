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
#ifndef BOOST_PYTHON_EIGEN_EIGEN_TO_PYTHON_HPP_INCLUDED
#define BOOST_PYTHON_EIGEN_EIGEN_TO_PYTHON_HPP_INCLUDED

#include <boost/python/numpy.hpp>
#include <Eigen/Core>

namespace boost { namespace python {

template <typename Matrix>
struct eigen_to_python {
    typedef typename Matrix::Scalar Scalar;
    typedef Eigen::Matrix<Scalar,Matrix::RowsAtCompileTime,
                          Matrix::ColsAtCompileTime,Eigen::RowMajor> TrueMatrix;

    BOOST_STATIC_CONSTANT(bool, shallow_possible = (Matrix::Flags & Eigen::DirectAccessBit));

    static object to_python_shallow(Matrix const & input, object const owner, bool writeable=true) {
        if (!shallow_possible) {
            PyErr_SetString(PyExc_TypeError, "Cannot convert matrix to Python with shared data.");
            throw_error_already_set();
        }
        numpy::dtype dtype = numpy::dtype::get_builtin<Scalar>();
        int itemsize = dtype.get_itemsize();
        std::vector<Py_intptr_t> shape(2);
        shape[0] = input.rows();
        shape[1] = input.cols();
        std::vector<Py_intptr_t> strides(2, itemsize);
        if (Matrix::Flags & Eigen::RowMajorBit) {
            strides[0] *= input.stride();
        } else {
            strides[1] *= input.stride();
        }
        if (writeable) {
            return numpy::matrix(
                numpy::from_data(input.data(), dtype, shape, strides, owner),
                dtype,
                false
            );
        } else {
            return numpy::matrix(
                numpy::from_data(const_cast<Scalar*>(input.data()), dtype, shape, strides, owner), 
                dtype,
                false
            );
        }
    }

    static object to_python_value(Matrix const & input) {
        numpy::dtype dtype = numpy::dtype::get_builtin<Scalar>();
        numpy::matrix matrix(
            numpy::zeros(make_tuple(input.rows(), input.cols()), dtype),
            dtype,
            false
        );
        Eigen::Map<TrueMatrix> map(reinterpret_cast<Scalar*>(matrix.get_data()), 
                                   matrix.shape(0), matrix.shape(1));
        map = input;
        return matrix;
    }

};

}} // namespace boost::python

#endif // !BOOST_PYTHON_EIGEN_EIGEN_TO_PYTHON_HPP_INCLUDED
