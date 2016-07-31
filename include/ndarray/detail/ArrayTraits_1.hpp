
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
#ifndef NDARRAY_detail_ArrayTraits_1_hpp_INCLUDED
#define NDARRAY_detail_ArrayTraits_1_hpp_INCLUDED

#include <type_traits>

#include "ndarray/common.hpp"
#include "ndarray/DType.hpp"

namespace ndarray {
namespace detail {

template <typename T, size_t N_>
struct ArrayTraits {
    typedef T element;
    static constexpr size_t N = N_;
    typedef get_dtype_t<T> dtype_t;
    typedef Layout<N> layout_t;
    typedef ArrayImpl<typename std::remove_const<T>::type,N> impl_t;
    typedef Array<T,N-1> value_type;
    typedef ArrayRef<T,N-1> reference;
    typedef Array<T,N-1> * pointer;
    typedef Iter<T,N> iterator;

    static reference make_reference_at(
        byte_t * buffer,
        ArrayBase<T,N> const & self
    );
};

template <typename T>
struct ArrayTraits<T,1> {
    typedef T element;
    static constexpr size_t N = 1;
    typedef get_dtype_t<T> dtype_t;
    typedef ArrayImpl<T,1> impl_t;
    typedef Layout<1> layout_t;
    typedef typename dtype_t::value_type value_type;
    typedef typename dtype_t::reference reference;
    typedef typename dtype_t::pointer pointer;
    typedef typename dtype_t::iterator iterator;

    static reference make_reference_at(
        byte_t * buffer,
        ArrayBase<T,N> const & self
    ) {
        return *reinterpret_cast<T*>(buffer);
    }
};

template <typename T>
struct ArrayTraits<T const,1> {
    typedef T const element;
    static constexpr size_t N = 1;
    typedef get_dtype_t<T const> dtype_t;
    typedef ArrayImpl<T,1> impl_t;
    typedef Layout<1> layout_t;
    typedef typename dtype_t::value_type value_type;
    typedef typename dtype_t::const_reference reference;
    typedef typename dtype_t::const_pointer pointer;
    typedef typename dtype_t::const_iterator iterator;

    static reference make_reference_at(
        byte_t * buffer,
        ArrayBase<T,N> const & self
    ) {
        return *reinterpret_cast<T const*>(buffer);
    }
};

} // namespace detail
} // namespace ndarray

#endif // !NDARRAY_detail_ArrayTraits_1_hpp_INCLUDED
