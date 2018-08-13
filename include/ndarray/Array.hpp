// -*- c++ -*-
/*
 * Copyright (c) 2010-2018, Jim Bosch
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

constexpr Offset nested_contiguousness(Size n, Offset c) {
    return (c > 0 && c == n) ? c - 1 : (c < 0 && -c == n) ? c + 1 : c;
}

} // namespace detail

template <typename T, Size N, Offset C> class Array;

template <typename Target> class Deref;

template <typename T, Offset C>
class Array<T const, 1, C> {
public:

    using Reference = T const &;

    Array() noexcept = default;

    template <Offset D>
    Array(Array<T const, 1, D> const & other) noexcept {
        static_assert(detail::contiguousness_convertible(1, D, C), "invalid contiguousness conversion");
    }

    template <Offset D>
    Array(Array<T const, 1, D> && other) noexcept {
        static_assert(detail::contiguousness_convertible(1, D, C), "invalid contiguousness conversion");
    }

    Array & operator=(Array const &) noexcept = default;

    Array & operator=(Array &&) noexcept = default;

    Deref<Array<T const, 1, C>> operator*() const noexcept;

    Reference operator[](Size n) const;

protected:
    detail::ArrayImpl<1> _impl;
};

template <typename T, Offset C>
class Array<T, 1, C> : public Array<T const, 1, C> {
public:

    using Reference = T &;

    Array() noexcept = default;

    template <Offset D>
    Array(Array<T, 1, D> const & other) noexcept {
        static_assert(detail::contiguousness_convertible(1, D, C), "invalid contiguousness conversion");
    }

    template <Offset D>
    Array(Array<T, 1, D> && other) noexcept {
        static_assert(detail::contiguousness_convertible(1, D, C), "invalid contiguousness conversion");
    }

    Array & operator=(Array const &) noexcept = default;

    Array & operator=(Array &&) noexcept = default;

    Deref<Array<T, 1, C>> operator*() const noexcept;

    Reference operator[](Size n) const;

};


template <typename T, Size N, Offset C>
class Array<T const, N, C> {
public:

    using Reference = Array<T const, N-1, detail::nested_contiguousness(N, C)>;

    Array() noexcept = default;

    Array(Array const &) noexcept = default;

    Array(Array &&) noexcept = default;

    template <Offset D>
    Array(Array<T const, N, D> const & other) noexcept {
        static_assert(detail::contiguousness_convertible(N, D, C), "invalid contiguousness conversion");
    }

    template <Offset D>
    Array(Array<T const, N, D> && other) noexcept {
        static_assert(detail::contiguousness_convertible(N, D, C), "invalid contiguousness conversion");
    }

    Array & operator=(Array const &) noexcept = default;

    Array & operator=(Array &&) noexcept = default;

    Deref<Array<T const, N, C>> operator*() const noexcept;

    Reference operator[](Size n) const;

protected:
    detail::ArrayImpl<1> _impl;
};

template <typename T, Size N, Offset C>
class Array : public Array<T const, N, C> {
public:

    using Reference = Array<T const, N-1, detail::nested_contiguousness(N, C)>;

    Array() noexcept = default;

    Array(Array const &) noexcept = default;

    Array(Array &&) noexcept = default;

    template <Offset D>
    Array(Array<T, N, D> const & other) noexcept {
        static_assert(detail::contiguousness_convertible(N, D, C), "invalid contiguousness conversion");
    }

    template <Offset D>
    Array(Array<T, N, D> && other) noexcept {
        static_assert(detail::contiguousness_convertible(N, D, C), "invalid contiguousness conversion");
    }

    Array & operator=(Array const &) noexcept = default;

    Array & operator=(Array &&) noexcept = default;

    Deref<Array<T, N, C>> operator*() const noexcept;

    Reference operator[](Size n) const;

};

} // namespace ndarray

#endif // !NDARRAY_ArrayBase_hpp_INCLUDED
