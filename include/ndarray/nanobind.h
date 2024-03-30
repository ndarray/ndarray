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

#ifndef NDARRAY_nanobind_h_INCLUDED
#define NDARRAY_nanobind_h_INCLUDED

/**
 *  @file ndarray/nanobind.h
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

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include "ndarray.h"
#include "ndarray/eigen.h"
#include "ndarray/Array.h"

#include <typeinfo>

namespace nb = nanobind;
namespace ndarray {
namespace detail {

    inline void destroyCapsule(PyObject *p) {
        void *m = PyCapsule_GetPointer(p, "ndarray.Manager");
        Manager::Ptr *b = reinterpret_cast<Manager::Ptr *>(m);
        delete b;
    }

} // namespace ndarray::detail

inline PyObject *makePyManager(Manager::Ptr const &m) {
    return PyCapsule_New(
            new Manager::Ptr(m),
            "ndarray.Manager",
            detail::destroyCapsule
    );
}

template<typename T, int N, int C>
struct
#ifdef __GNUG__
// pybind11 hides all symbols in its namespace only when this is set,
// and in that case we should hide these classes too.
            __attribute__((visibility("hidden")))
#endif
NanobindHelper {
};

} // namespace ndarray
NAMESPACE_BEGIN(NB_NAMESPACE)
NAMESPACE_BEGIN(detail)
template<typename T, int N, int C>
struct type_caster<::ndarray::Array<T,N,C>> {
    using Wrapper = std::remove_const_t<nb::ndarray<nb::numpy, T>>;
    using ArrayType = nb::ndarray<nb::numpy, typename std::remove_const_t<T>> ;
    using Array = std::conditional_t<std::is_const_v<T>, const ArrayType, ArrayType>;
    using Element =  typename std::remove_const_t<T>;
    static constexpr bool isConst = std::is_const<Element>::value;

    using Value = ::ndarray::Array<T,N,C>;
    static constexpr auto Name = const_name("ndarray");
    template<typename T_> using Cast = movable_cast_t<T_>;

    static handle from_cpp(Value *p, rv_policy policy, cleanup_list *list) {
        if (!p)return none().release();
        return from_cpp(*p, policy, list);
    }

    bool init(nb::handle src, cleanup_list *cleanup) {
        isNone = src.is_none();
        if (isNone) {
            return true;
        }

        int64_t shape[N];
        ndarray_config config;
        config.shape = shape;
        wrapper = Wrapper(ndarray_import(
                src.ptr(), &config, true, cleanup));
        return wrapper.is_valid();
    }

    bool check() const {
        if (isNone) {
            return true;
        }

        if (wrapper.ndim() != N) {
            return false;
        }
        if(wrapper.dtype().bits != sizeof(Element) * 8) {
            return false;
        }
        switch(dlpack::dtype_code(wrapper.dtype().code)) {
            case dlpack::dtype_code::Float:
                if(!std::is_floating_point_v<Element>) return false;
                break;
            case dlpack::dtype_code::Int:
                if(!(std::is_signed_v<Element> && std::is_integral_v<Element>)) return false;
                break;
            case dlpack::dtype_code::UInt:
                if(!(std::is_unsigned_v<Element> && std::is_integral_v<Element>)) return false;
                break;
            case dlpack::dtype_code::Bool:
                if(!std::is_same_v<Element, bool>) return false;
                break;
            default:
                return false;
        }

        //if (!isConst && !wrapper.writeable()) {
        //    return false;
        //}

        int64_t const * shape = wrapper.shape_ptr();
        int64_t const * strides = wrapper.stride_ptr();
        size_t const itemsize = wrapper.itemsize();
        if (C > 0) {
            // If the shape is zero in any dimension, we don't
            // worry about the strides.
            for (int i = 0; i < C; ++i) {
                if (shape[N-i-1] == 0) {
                    return true;
                }
            }

            int64_t requiredStride = 1;//itemsize;
            for (int i = 0; i < C; ++i) {
                if (strides[N-i-1] != requiredStride) {
                    return false;
                }
                requiredStride *= shape[N-i-1];
            }
        } else if (C < 0) {
            // If the shape is zero in any dimension, we don't
            // worry about the strides.
            for (int i = 0; i < -C; ++i) {
                if (shape[i] == 0) {
                    return true;
                }
            }
            size_t requiredStride = itemsize;
            for (int i = 0; i < -C; ++i) {
                if (strides[i] != requiredStride) {
                    return false;
                }
                requiredStride *= shape[i];
            }
        }
        return true;
    }

    Value convert() const {
        if (isNone) {
            return Value();
        }

        //if (!wrapper.dtype().attr("isnative")) {
        //    throw nb::type_error("Only arrays with native byteorder can be converted to C++.");
        //}

        ::ndarray::Vector<::ndarray::Size,N> nShape;
        ::ndarray::Vector<::ndarray::Offset,N> nStrides;
        int64_t const * pShape = wrapper.shape_ptr();
        int64_t const * pStrides = wrapper.stride_ptr();
        size_t itemsize = wrapper.itemsize();
        for (int i = 0; i < N; ++i) {
            nShape[i] = pShape[i];
            nStrides[i] = pStrides[i];
        }

        auto *p = const_cast<Element*>(wrapper.data());

        return Value (
                ::ndarray::external(const_cast<Element*>(wrapper.data()),
                                  nShape, nStrides, wrapper)
        );
    }

    void set_value() {
        value = convert();
    }

    explicit operator Value * () {
        if (isNone) {
            return nullptr;
        } else {
            set_value();
            return &value;
        }
    }

    explicit operator Value &() {
        set_value();
        return (Value &) value;
    }

    explicit operator Value &&() {
        set_value();
        return (Value &&) value;
    }


    bool from_python(nb::handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
        bool result = init(src, cleanup) && check();
        return result;
    }
    static nb::handle from_cpp(const ::ndarray::Array<T, N, C> &src, rv_policy policy,
                               cleanup_list *cleanup) noexcept {
        using ArrayType = nb::ndarray<nb::numpy, typename std::remove_const_t<T>> ;
        using Array = std::conditional_t<std::is_const_v<T>, const ArrayType, ArrayType>;
        using Element =  typename std::remove_const_t<T>;
        ::ndarray::Vector<::ndarray::Size,N> nShape = src.getShape();
        ::ndarray::Vector<::ndarray::Offset,N> nStrides = src.getStrides();
        std::vector<size_t> pShape(N);
        std::vector<int64_t> pStrides(N);
        for (int i = 0; i < N; ++i) {
            pShape[i] = nShape[i];
            pStrides[i] = nStrides[i];
        }
        nb::object base = nb::object();
        if (src.getManager()) {
            base = nb::steal<nb::object>(::ndarray::makePyManager(src.getManager()));
        }
        Array array((Element*)src.getData(), N, pShape.data(), base, pStrides.data());

        nb::handle result = ndarray_export(array.handle(), nb::numpy::value, policy, cleanup);
        if (std::is_const_v<T>) {
            result.attr("flags")["WRITEABLE"] = false;
        }
        return result;
    }
private:
    bool isNone = false;
    Value value;
    Wrapper wrapper = Wrapper();
};
NAMESPACE_END(detail);
NAMESPACE_END(NB_NAMESPACE);
#endif
