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
#ifndef NDARRAY_Array_1_hpp_INCLUDED
#define NDARRAY_Array_1_hpp_INCLUDED

#include "ndarray/common.hpp"
#include "ndarray/detail/ArrayTraits_1.hpp"
#include "ndarray/detail/ArrayImpl.hpp"
#include "ndarray/detail/ArrayBase_1.hpp"

namespace ndarray {

template <typename T, int N>
class Array<T const,N> : public ArrayBase<T const,N> {
    typedef ArrayBase<T const,N> base_t;
    typedef detail::ArrayTraits<T const,N> traits_t;
    typedef typename traits_t::impl_t impl_t;
    template <typename U, int M> friend struct detail::ArrayTraits;
public:

    Array() : base_t() {}

    Array(Array const &) = default;

    Array(Array &&) = default;

    Array & operator=(Array const &) = default;

    Array & operator=(Array &&) = default;

    bool operator==(ArrayBase<T,N> const & other) const {
        return this->_impl == other._impl;
    }

    bool operator!=(ArrayBase<T,N> const & other) const {
        return this->_impl != other._impl;
    }

#ifdef NDARRAY_FAST_CONVERSIONS
    ArrayRef<T const,N> const & operator*() const;
#else
    ArrayRef<T const,N> operator*() const;
#endif

protected:
    explicit Array(impl_t const & impl) : base_t(impl) {}
    explicit Array(impl_t && impl) : base_t(std::move(impl)) {}
};

template <typename T, int N>
class Array : public ArrayBase<T,N> {
    typedef ArrayBase<T,N> base_t;
    typedef detail::ArrayTraits<T,N> traits_t;
    typedef typename traits_t::impl_t impl_t;
    template <typename U, int M> friend struct detail::ArrayTraits;
public:

    typedef typename base_t::dtype_t dtype_t;

    Array() : base_t() {}

    template <typename ShapeVector>
    Array(
        ShapeVector const & shape,
        MemoryOrder order=MemoryOrder::ROW_MAJOR,
        dtype_t const & dtype=dtype_t()
    ) :
        base_t(impl_t(shape, order, dtype))
    {}

    template <typename ShapeVector, typename StridesVector>
    Array(
        ShapeVector const & shape,
        StridesVector const & strides,
        dtype_t const & dtype=dtype_t()
    ) :
        base_t(impl_t(shape, strides, dtype))
    {}

    Array(Array const &) = default;

    Array(Array &&) = default;

    Array & operator=(Array const &) = default;

    Array & operator=(Array &&) = default;

    bool operator==(ArrayBase<T const,N> const & other) const {
        return this->_impl == other._impl;
    }

    bool operator!=(ArrayBase<T const,N> const & other) const {
        return this->_impl != other._impl;
    }

#ifdef NDARRAY_FAST_CONVERSIONS
    ArrayRef<T,N> const & operator*() const;
#else
    ArrayRef<T,N> operator*() const;
#endif

#ifdef NDARRAY_FAST_CONVERSIONS
    operator Array<T const,N> const & () const;
#else
    operator Array<T const,N>() const;
#endif

protected:
    explicit Array(impl_t const & impl) : base_t(impl) {}
    explicit Array(impl_t && impl) : base_t(std::move(impl)) {}
};

} // namespace ndarray

#endif // !NDARRAY_Array_1_hpp_INCLUDED
