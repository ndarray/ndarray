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

} // namespace detail

template <typename T, Size N, Offset C> class Array;

template <typename T, Size N, Offset C> class ArrayRef;

template <typename T, Size N, Offset C>
class Array<T const, N, C> {
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

    template <Offset D>
    Array(Array<T const, N, D> const & other) {
        static_assert(detail::contiguousness_convertible(N, D, C), "invalid contiguousness conversion");
    }

    template <Offset D>
    Array(Array<T const, N, D> && other) {
        static_assert(detail::contiguousness_convertible(N, D, C), "invalid contiguousness conversion");
    }

    Array & operator=(Array const &) = default;

    Array & operator=(Array &&) = default;

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

    Array & operator=(Array const &) = default;

    Array & operator=(Array &&) = default;

};

template <typename T, Size N, Offset C>
class ArrayRef<T const, N, C> : public Array<T const, N, C> {
    using Base = Array<T const, N, C>;
public:

    ArrayRef() = default;

    template <Offset D>
    ArrayRef(Array<T, N, D> const & other) : Base(other) {}

    template <Offset D>
    ArrayRef(Array<T, N, D> && other) : Base(std::move(other)) {}

    template <Offset D>
    ArrayRef(Array<T const, N, D> const & other) : Base(other) {}

    template <Offset D>
    ArrayRef(Array<T const, N, D> && other) : Base(std::move(other)) {}

    ArrayRef & operator=(ArrayRef const &) = delete;

    ArrayRef & operator=(ArrayRef &&) = delete;

    template <typename U, Offset D>
    ArrayRef const & operator=(Array<U, N, D> const & other) const = delete;

};

template <typename T, Size N, Offset C>
class ArrayRef : public Array<T, N, C> {
    using Base = Array<T, N, C>;
public:

    ArrayRef() = default;

    template <Offset D>
    ArrayRef(Array<T, N, D> const & other) : Base(other) {}

    template <Offset D>
    ArrayRef(Array<T, N, D> && other) : Base(std::move(other)) {}

    ArrayRef const & operator=(ArrayRef const &) const;

    ArrayRef const & operator=(ArrayRef &&) const;

    template <typename U, Offset D>
    ArrayRef const & operator=(Array<U, N, D> const & other) const;

};

} // namespace ndarray

#endif // !NDARRAY_ArrayBase_hpp_INCLUDED
