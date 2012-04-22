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
#ifndef NDARRAY_EIGEN_BP_to_python_h_INCLUDED
#define NDARRAY_EIGEN_BP_to_python_h_INCLUDED

#include "boost/numpy.hpp"
#include "ndarray/eigen.h"
#include <Eigen/Core>

namespace boost { namespace python {

template <typename Input> struct eigen_to_python;

template <typename T, int N, int C, typename XprKind_, int Rows_, int Cols_>


template <typename Input>
struct eigen_to_python {
    typedef typename Input::Scalar Scalar;
    typedef typename ndarray::SelectEigenView<Input>::Type View;

    BOOST_STATIC_CONSTANT(bool, shallow_possible = (Input::Flags & Eigen::DirectAccessBit));

    static numpy::ndarray to_python_shallow(Input const & input, object const owner, bool writeable=true) {
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
        strides[0] *= input.rowStride();
        strides[1] *= input.colStride();
        numpy::ndarray result = (writeable) ?
            numpy::from_data(const_cast<Scalar const*>(input.data()), dtype, shape, strides, owner) :
            numpy::from_data(const_cast<Scalar*>(input.data()), dtype, shape, strides, owner);
        if (Input::Flags & Eigen::IsVectorAtCompileTime) result = result.squeeze();
        return result;
    }

    static numpy::ndarray to_python_value(Input const & input) {
        View tmp = ndarray::copy(input);
        return eigen_to_python<View>::to_python_shallow(input);
    }

};

}} // namespace boost::python

#endif // !NDARRAY_EIGEN_BP_to_python_h_INCLUDED
