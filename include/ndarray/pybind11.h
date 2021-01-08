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
#include "pybind11/numpy.h"

#include "ndarray.h"
#include "ndarray/eigen.h"

namespace ndarray {

namespace detail {

inline void destroyCapsule(PyObject * p) {
    void * m = PyCapsule_GetPointer(p, "ndarray.Manager");
    Manager::Ptr * b = reinterpret_cast<Manager::Ptr*>(m);
    delete b;
}

} // namespace ndarray::detail

inline PyObject* makePyManager(Manager::Ptr const & m) {
    return PyCapsule_New(
        new Manager::Ptr(m),
        "ndarray.Manager",
        detail::destroyCapsule
    );
}

#if PYBIND11_VERSION_MAJOR == 2 && PYBIND11_VERSION_MINOR <= 1
using pybind11_np_size_t = size_t;
#else
using pybind11_np_size_t = ssize_t;
#endif

template <typename T, int N, int C>
struct
#ifdef __GNUG__
// pybind11 hides all symbols in its namespace only when this is set,
// and in that case we should hide these classes too.
__attribute__((visibility("hidden")))
#endif
Pybind11Helper {
    using Element = typename ndarray::Array<T,N,C>::Element;
    using Wrapper = pybind11::array_t<typename std::remove_const<Element>::type, 0>;  // 0: no ensurecopy
    static constexpr bool isConst = std::is_const<Element>::value;

    Pybind11Helper() : isNone(false), wrapper() {}

    bool init(pybind11::handle src) {
        isNone = src.is_none();
        if (isNone) {
            return true;
        }
        if (!Wrapper::check_(src)) {
            return false;
        }
        try {
            wrapper = pybind11::reinterpret_borrow<Wrapper>(src);
        } catch (pybind11::error_already_set & err) {
            return false;
        }
        return true;
    }

    bool check() const {
        if (isNone) {
            return true;
        }
        if (!wrapper) {
            return false;
        }
        if (wrapper.ndim() != N) {
            return false;
        }
        if (!isConst && !wrapper.writeable()) {
            return false;
        }
        pybind11_np_size_t const * shape = wrapper.shape();
        pybind11_np_size_t const * strides = wrapper.strides();
        pybind11_np_size_t const itemsize = wrapper.itemsize();
        if (C > 0) {
            pybind11_np_size_t requiredStride = itemsize;
            for (int i = 0; i < C; ++i) {
                if (strides[N-i-1] != requiredStride) {
                    return false;
                }
                requiredStride *= shape[N-i-1];
            }
        } else if (C < 0) {
            pybind11_np_size_t requiredStride = itemsize;
            for (int i = 0; i < -C; ++i) {
                if (strides[i] != requiredStride) {
                    return false;
                }
                requiredStride *= shape[i];
            }
        }
        return true;
    }

    ndarray::Array<T,N,C> convert() const {
        if (isNone) {
            return ndarray::Array<T,N,C>();
        }
        if (!pybind11::reinterpret_borrow<pybind11::bool_>(wrapper.dtype().attr("isnative"))) {
            PyErr_SetString(PyExc_TypeError, "Only arrays with native byteorder can be converted to C++.");
            throw pybind11::error_already_set();
        }
        Vector<ndarray::Size,N> nShape;
        Vector<ndarray::Offset,N> nStrides;
        pybind11_np_size_t const * pShape = wrapper.shape();
        pybind11_np_size_t const * pStrides = wrapper.strides();
        pybind11_np_size_t const itemsize = wrapper.itemsize();
        for (int i = 0; i < N; ++i) {
            if (pStrides[i] % itemsize != 0) {
                PyErr_SetString(
                    PyExc_TypeError,
                    "Cannot convert array to C++: strides must be an integer multiple of the element size"
                );
                throw pybind11::error_already_set();
            }
            nShape[i] = pShape[i];
            nStrides[i] = pStrides[i]/itemsize;
        }
        return ndarray::Array<T,N,C>(
            ndarray::external(const_cast<Element*>(wrapper.data()),
                              nShape, nStrides, pybind11::object(wrapper))
        );
    }

    static pybind11::handle toPython(ndarray::Array<T,N,C> const & src) {
        Vector<Size,N> nShape = src.getShape();
        Vector<Offset,N> nStrides = src.getStrides();
        std::vector<pybind11_np_size_t> pShape(N);
        std::vector<pybind11_np_size_t> pStrides(N);
        pybind11_np_size_t const itemsize = sizeof(Element);
        for (int i = 0; i < N; ++i) {
            pShape[i] = nShape[i];
            pStrides[i] = nStrides[i]*itemsize;
        }
        pybind11::object base;
        if (src.getManager()) {
            base = pybind11::reinterpret_steal<pybind11::object>(ndarray::makePyManager(src.getManager()));
        }
        Wrapper result(pShape, pStrides, src.getData(), base);
        if (std::is_const<Element>::value) {
            result.attr("flags")["WRITEABLE"] = false;
        }
        return result.release();
    }

    bool isNone;
    Wrapper wrapper;
};

} // namespace ndarray

namespace pybind11 {
namespace detail {

/* @brief A pybind11 type_caster for ndarray::Array
 */
template <typename T, int N, int C>
class type_caster< ndarray::Array<T,N,C> > {
    using Helper = ndarray::Pybind11Helper<T,N,C>;
public:

    bool load(handle src, bool) {
        return _helper.init(src) && _helper.check();
    }

    void set_value() {
        _value = _helper.convert();
    }

    static handle cast(const ndarray::Array<T,N,C> &src, return_value_policy /* policy */, handle /* parent */) {
        return Helper::toPython(src);
    }

    static constexpr auto name = _("numpy.ndarray");

    static handle cast(const ndarray::Array<T,N,C> *src, return_value_policy policy, handle parent) {
        return cast(*src, policy, parent);
    }

    operator ndarray::Array<T,N,C> * () {
        if (_helper.isNone) {
            return nullptr;
        } else {
            set_value();
            return &_value;
        }
    }

    operator ndarray::Array<T,N,C> & () { set_value(); return _value; }

    template <typename _T> using cast_op_type = pybind11::detail::cast_op_type<_T>;

private:
    ndarray::Array<T,N,C> _value;
    Helper _helper;
};

} // namespace detail
} // namespace pybind11

#endif // !NDARRAY_pybind11_h_INCLUDED
