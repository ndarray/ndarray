/*
 * LSST Data Management System
 * Copyright 2008-2016  AURA/LSST.
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
 * see <https://www.lsstcorp.org/LegalNotices/>.
 */

#ifndef NDARRAY_pybind11_h_INCLUDED
#define NDARRAY_pybind11_h_INCLUDED

/**
 *  @file ndarray/pybind11.h
 *  @brief Public header file for pybind11-based Python support.
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

#include "pybind11.h"

#include "Python.h"
#include "ndarray.h"
#include "ndarray/converter/numpy.h"
#include "ndarray/converter/ufunctors.h"
#include "ndarray/converter/Vector.h"
#include "ndarray/converter/eigen.h"

NAMESPACE_BEGIN(pybind11)
NAMESPACE_BEGIN(detail)

/* @brief A pybind11 type_caster for ndarray::Array
 */
template <typename T, int N, int C>
class type_caster< ndarray::Array<T,N,C> > {
public:
    bool load(handle src, bool) {
        _src.reset(src.ptr()); // equivalent to add_ref=true, keep alive for stage 2
        if (!ndarray::PyConverter< ndarray::Array<T,N,C> >::fromPythonStage1(_src)) {
            PyErr_Clear();
            return false;
        }
        return true;
    }
    static handle cast(const ndarray::Array<T,N,C> &src, return_value_policy /* policy */, handle /* parent */) {
        return ndarray::PyConverter< ndarray::Array<T,N,C> >::toPython(src);
    }
    void set_value() {
        if (!ndarray::PyConverter< ndarray::Array<T,N,C> >::fromPythonStage2(_src, _value)) {
            throw error_already_set();
        }
    }
/* This part is normally created by the PYBIND11_TYPE_CASTER macro, which
 * can't be used here due to the partial specialization
 */
protected:
    ndarray::PyPtr _src;
    ndarray::Array<T,N,C> _value;
public:
    static PYBIND11_DESCR name() { return type_descr(_<ndarray::Array<T,N,C>>()); }
    static handle cast(const ndarray::Array<T,N,C> *src, return_value_policy policy, handle parent) {
        return cast(*src, policy, parent);
    }
    operator ndarray::Array<T,N,C> * () { set_value(); return &_value; }
    operator ndarray::Array<T,N,C> & () { set_value(); return _value; }
    template <typename _T> using cast_op_type = pybind11::detail::cast_op_type<_T>;
};

/* @brief A pybind11 type_caster for ndarray::EigenView
 */
template <typename T, int N, int C, typename Kind, int Rows, int Cols>
class type_caster< ndarray::EigenView<T,N,C,Kind,Rows,Cols> > {
public:
    bool load(handle src, bool) {
        _src.reset(src.ptr()); // equivalent to add_ref=true, keep alive for stage 2
        if (!ndarray::PyConverter< ndarray::EigenView<T,N,C,Kind,Rows,Cols> >::fromPythonStage1(_src)) {
            PyErr_Clear();
            return false;
        }
        return true;
    }
    static handle cast(const ndarray::EigenView<T,N,C,Kind,Rows,Cols> &src, return_value_policy /* policy */, handle /* parent */) {
        return ndarray::PyConverter< ndarray::EigenView<T,N,C,Kind,Rows,Cols> >::toPython(src);
    }
    void set_value() {
        if (!ndarray::PyConverter< ndarray::EigenView<T,N,C,Kind,Rows,Cols> >::fromPythonStage2(_src, _value)) {
            throw error_already_set();
        }
    }
/* This part is normally created by the PYBIND11_TYPE_CASTER macro, which
 * can't be used here due to the partial specialization
 */
protected:
    ndarray::PyPtr _src;
    ndarray::EigenView<T,N,C,Kind,Rows,Cols> _value;
public:
    static PYBIND11_DESCR name() { return type_descr(_<ndarray::EigenView<T,N,C,Kind,Rows,Cols>>()); }
    static handle cast(const ndarray::EigenView<T,N,C,Kind,Rows,Cols> *src, return_value_policy policy, handle parent) {
        return cast(*src, policy, parent);
    }
    operator ndarray::EigenView<T,N,C,Kind,Rows,Cols> * () { set_value(); return &_value; }
    operator ndarray::EigenView<T,N,C,Kind,Rows,Cols> & () { set_value(); return _value; }
    template <typename _T> using cast_op_type = pybind11::detail::cast_op_type<_T>;
};

/* @brief A pybind11 type_caster for Eigen::Array
 */
template<typename T, int R, int C, int O, int MR, int MC>
class type_caster< Eigen::Array<T,R,C,O,MR,MC> > {
public:
    bool load(handle src, bool) {
        _src.reset(src.ptr()); // equivalent to add_ref=true, keep alive for stage 2
        if (!ndarray::PyConverter< Eigen::Array<T,R,C,O,MR,MC> >::fromPythonStage1(_src)) {
            PyErr_Clear();
            return false;
        }
        return true;
    }
    static handle cast(const Eigen::Array<T,R,C,O,MR,MC> &src, return_value_policy /* policy */, handle /* parent */) {
        return ndarray::PyConverter< Eigen::Array<T,R,C,O,MR,MC> >::toPython(src);
    }
    void set_value() {
        if (!ndarray::PyConverter< Eigen::Array<T,R,C,O,MR,MC> >::fromPythonStage2(_src, _value)) {
            throw error_already_set();
        }
    }
/* This part is normally created by the PYBIND11_TYPE_CASTER macro, which
 * can't be used here due to the partial specialization
 */
protected:
    ndarray::PyPtr _src;
    Eigen::Array<T,R,C,O,MR,MC> _value;
public:
    static PYBIND11_DESCR name() { return type_descr(_<Eigen::Array<T,R,C,O,MR,MC>>()); }
    static handle cast(const Eigen::Array<T,R,C,O,MR,MC> *src, return_value_policy policy, handle parent) {
        return cast(*src, policy, parent);
    }
    operator Eigen::Array<T,R,C,O,MR,MC> * () { set_value(); return &_value; }
    operator Eigen::Array<T,R,C,O,MR,MC> & () { set_value(); return _value; }
    template <typename _T> using cast_op_type = pybind11::detail::cast_op_type<_T>;
};

/* @brief A pybind11 type_caster for Eigen::Matrix
 */
template<typename T, int R, int C, int O, int MR, int MC>
class type_caster< Eigen::Matrix<T,R,C,O,MR,MC> > {
public:
    bool load(handle src, bool) {
        _src.reset(src.ptr()); // equivalent to add_ref=true, keep alive for stage 2
        if (!ndarray::PyConverter< Eigen::Matrix<T,R,C,O,MR,MC> >::fromPythonStage1(_src)) {
            PyErr_Clear();
            return false;
        }
        return true;
    }
    static handle cast(const Eigen::Matrix<T,R,C,O,MR,MC> &src, return_value_policy /* policy */, handle /* parent */) {
        return ndarray::PyConverter< Eigen::Matrix<T,R,C,O,MR,MC> >::toPython(src);
    }
    void set_value() {
        if (!ndarray::PyConverter< Eigen::Matrix<T,R,C,O,MR,MC> >::fromPythonStage2(_src, _value)) {
            throw error_already_set();
        }
    }
/* This part is normally created by the PYBIND11_TYPE_CASTER macro, which
 * can't be used here due to the partial specialization
 */
protected:
    ndarray::PyPtr _src;
    Eigen::Matrix<T,R,C,O,MR,MC> _value;
public:
    static PYBIND11_DESCR name() { return type_descr(_<Eigen::Matrix<T,R,C,O,MR,MC>>()); }
    static handle cast(const Eigen::Matrix<T,R,C,O,MR,MC> *src, return_value_policy policy, handle parent) {
        return cast(*src, policy, parent);
    }
    operator Eigen::Matrix<T,R,C,O,MR,MC> * () { set_value(); return &_value; }
    operator Eigen::Matrix<T,R,C,O,MR,MC> & () { set_value(); return _value; }
    template <typename _T> using cast_op_type = pybind11::detail::cast_op_type<_T>;
};

NAMESPACE_END(detail)
NAMESPACE_END(pybind11)

#endif // !NDARRAY_pybind11_h_INCLUDED
