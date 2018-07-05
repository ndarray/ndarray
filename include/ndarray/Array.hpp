// -*- c++ -*-
/*
 * Copyright (c) 2010-2016, Jim Bosch
 * All rights reserved.
 *
 * ndarray is distributed under a simple BSD-like license;
 * see the LICENSE file that should be present in the root
 * of the source distribution, or alternately available at:
 * https://github.com/ndarray/ndarray
 */
#ifndef NDARRAY_ArrayBase_hpp_INCLUDED
#define NDARRAY_ArrayBase_hpp_INCLUDED

#include "ndarray/common.hpp"
#include "ndarray/detail/ArrayImpl.hpp"

namespace ndarray {

    namespace detail {

    constexpr bool contiguousness_convertible(Size n, Offset in, Offset out) {
        return (in >= 0 && out >= 0 && in >= out) ||
               (in <= 0 && out <= 0 && in <= out) ||
               (n == 1 && (in == 1 || in == -1) && (out == 1 || out == -1));
    }

    template <typename T_in, typename T_out, Size N, Offset C_in, Offset C_out>
    using EnableIfConvertible = std::enable_if_t<
            std::is_convertible<T_in*, T_out*>::value &&
            contiguousness_convertible(N, C_in, C_out)
        >;

    } // namespace detail

template <typename T, Size N, Offset C> class Array;

template <typename T, Size N, Offset C>
class Array<T const, N, C> {
public:

    Array() = default;

    template <typename U, Offset D>
    Array(Array<U, N, D> const & other) {
        static_assert(std::is_convertible<U*, T const*>::value, "invalid pointer conversion");
        static_assert(detail::contiguousness_convertible(N, D, C), "invalid contiguousness conversion");
    }

    template <typename U, Offset D>
    Array(Array<U, N, D> && other) {
        static_assert(std::is_convertible<U*, T const*>::value, "invalid pointer conversion");
        static_assert(detail::contiguousness_convertible(N, D, C), "invalid contiguousness conversion");
    }

};

template <typename T, Size N, Offset C>
class Array {
public:

    Array() = default;

    template <Offset D>
    Array(Array<T, N, D> const & other) {
        static_assert(detail::contiguousness_convertible(N, D, C), "invalid contiguousness conversion");
    }

    template <Offset D>
    Array(Array<T, N, D> && other) {
        static_assert(detail::contiguousness_convertible(N, D, C), "invalid contiguousness conversion");
    }

};

} // namespace ndarray

#endif // !NDARRAY_ArrayBase_hpp_INCLUDED
