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
#ifndef NDARRAY_ArrayRef_1_hpp_INCLUDED
#define NDARRAY_ArrayRef_1_hpp_INCLUDED

#include "ndarray/common.hpp"
#include "ndarray/detail/ArrayTraits_1.hpp"
#include "ndarray/detail/ArrayImpl.hpp"
#include "ndarray/detail/Array_1.hpp"

namespace ndarray {

template <typename T, size_t N, offset_t C>
class ArrayRef<T const,N,C> : public Array<T const,N,C> {
    typedef Array<T const,N,C> base_t;
    typedef detail::ArrayTraits<T const,N,C> traits_t;
    typedef typename traits_t::impl_t impl_t;
    template <typename U, size_t M, offset_t D> friend struct detail::ArrayTraits;
    template <typename U, size_t M, offset_t D> friend class Array;
public:

    ArrayRef() : base_t() {}

    ArrayRef(ArrayRef const &) = default;

    ArrayRef(ArrayRef &&) = default;

    ArrayRef & operator=(ArrayRef const &) = delete;

    ArrayRef & operator=(ArrayRef &&) = delete;

    Array<T const,N,C> & shallow() { return *this; }

#ifdef NDARRAY_FAST_CONVERSIONS

    template <offset_t D>
    operator ArrayRef<T const,N,D> const & () const;

#else

    template <offset_t D>
    operator ArrayRef<T const,N,D>() const;

#endif

private:
    explicit ArrayRef(impl_t const & impl) : base_t(impl) {}
    explicit ArrayRef(impl_t && impl) : base_t(std::move(impl)) {}
};

template <typename T, size_t N, offset_t C>
class ArrayRef : public Array<T,N,C> {
    typedef Array<T,N,C> base_t;
    typedef detail::ArrayTraits<T,N,C> traits_t;
    typedef typename traits_t::impl_t impl_t;
    template <typename U, size_t M, offset_t D> friend struct detail::ArrayTraits;
    template <typename U, size_t M, offset_t D> friend class Array;
public:

    typedef typename base_t::dtype_t dtype_t;

    ArrayRef() : base_t() {}

    ArrayRef(ArrayRef const &) = default;

    ArrayRef(ArrayRef &&) = default;

    ArrayRef & operator=(ArrayRef const &) = delete;

    ArrayRef & operator=(ArrayRef &&) = delete;

    ArrayRef const & operator=(Array<T const,N,C> const & other) const;

    ArrayRef const & operator=(Array<T,N,C> && other) const;

    Array<T,N,C> & shallow() { return *this; }

#ifdef NDARRAY_FAST_CONVERSIONS

    template <offset_t D>
    operator ArrayRef<T,N,D> const & () const;

    template <offset_t D>
    operator ArrayRef<T const,N,D> const & () const;

#else

    template <offset_t D>
    operator ArrayRef<T,N,D>() const;

    template <offset_t D>
    operator ArrayRef<T const,N,D>() const;

#endif

    ArrayRef const & operator=(T scalar) const;

private:
    explicit ArrayRef(impl_t const & impl) : base_t(impl) {}
    explicit ArrayRef(impl_t && impl) : base_t(std::move(impl)) {}
};

} // namespace ndarray

#endif // !NDARRAY_ArrayRef_1_hpp_INCLUDED
