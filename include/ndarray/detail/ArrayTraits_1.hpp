
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

inline constexpr offset_t compute_contiguousness(size_t N, offset_t C) {
    return (static_cast<offset_t>(N) == C) ? (C - 1) : ((C > 0) ? C : 0);
}

template <typename T, size_t N_, offset_t C_>
struct ArrayTraits {
    typedef T element;
    static constexpr size_t N = N_;
    static constexpr offset_t C = C_;
    static constexpr offset_t nestedC = compute_contiguousness(N, C);
    typedef get_dtype_t<T> dtype_t;
    typedef Layout<N> layout_t;
    typedef ArrayImpl<typename std::remove_const<T>::type,N> impl_t;
    typedef Array<T,N-1,nestedC> value_type;
    typedef ArrayRef<T,N-1,nestedC> reference;
    typedef Array<T,N-1,nestedC> * pointer;
    typedef Iter<T,N,C> iterator;

    static reference make_reference_at(
        byte_t * buffer,
        ArrayBase<T,N,C> const & self
    );

    static iterator make_iterator_at(
        byte_t * buffer,
        ArrayBase<T,N,C> const & parent
    );
};

template <typename T>
struct ArrayTraits<T,1,1> {
    typedef T element;
    static constexpr size_t N = 1;
    static constexpr offset_t C = 1;
    typedef get_dtype_t<T> dtype_t;
    typedef ArrayImpl<T,1> impl_t;
    typedef Layout<1> layout_t;
    typedef typename dtype_t::value_type value_type;
    typedef typename dtype_t::reference reference;
    typedef typename dtype_t::pointer pointer;
    typedef pointer iterator;

    static reference make_reference_at(
        byte_t * buffer,
        ArrayBase<T,1,1> const & self
    ) {
        return *reinterpret_cast<T*>(buffer);
    }

    static iterator make_iterator_at(
        byte_t * buffer,
        ArrayBase<T,1,1> const &
    ) {
        return reinterpret_cast<T*>(buffer);
    }

};

template <typename T>
struct ArrayTraits<T,1,-1> {
    typedef T element;
    static constexpr size_t N = 1;
    static constexpr offset_t C = 1;
    typedef get_dtype_t<T> dtype_t;
    typedef ArrayImpl<T,1> impl_t;
    typedef Layout<1> layout_t;
    typedef typename dtype_t::value_type value_type;
    typedef typename dtype_t::reference reference;
    typedef typename dtype_t::pointer pointer;
    typedef pointer iterator;

    static reference make_reference_at(
        byte_t * buffer,
        ArrayBase<T,1,-1> const & self
    ) {
        return *reinterpret_cast<T*>(buffer);
    }

    static iterator make_iterator_at(
        byte_t * buffer,
        ArrayBase<T,1,-1> const & parent
    ) {
        return reinterpret_cast<T*>(buffer);
    }

};

template <typename T>
struct ArrayTraits<T,1,0> {
    typedef T element;
    static constexpr size_t N = 1;
    static constexpr offset_t C = 0;
    typedef get_dtype_t<T> dtype_t;
    typedef ArrayImpl<T,1> impl_t;
    typedef Layout<1> layout_t;
    typedef typename dtype_t::value_type value_type;
    typedef typename dtype_t::reference reference;
    typedef typename dtype_t::pointer pointer;
    typedef Iter<T,1,0> iterator;

    static reference make_reference_at(
        byte_t * buffer,
        ArrayBase<T,1,0> const & self
    ) {
        return *reinterpret_cast<T*>(buffer);
    }

    static iterator make_iterator_at(
        byte_t * buffer,
        ArrayBase<T,1,0> const & parent
    );

};

template <typename T>
struct ArrayTraits<T const,1,1> {
    typedef T const element;
    static constexpr size_t N = 1;
    static constexpr offset_t C = 1;
    typedef get_dtype_t<T const> dtype_t;
    typedef ArrayImpl<T,1> impl_t;
    typedef Layout<1> layout_t;
    typedef typename dtype_t::value_type value_type;
    typedef typename dtype_t::const_reference reference;
    typedef typename dtype_t::const_pointer pointer;
    typedef pointer iterator;

    static reference make_reference_at(
        byte_t * buffer,
        ArrayBase<T const,1,1> const & self
    ) {
        return *reinterpret_cast<T const*>(buffer);
    }

    static iterator make_iterator_at(
        byte_t * buffer,
        ArrayBase<T const,1,1> const &
    ) {
        return reinterpret_cast<T*>(buffer);
    }

};

template <typename T>
struct ArrayTraits<T const,1,-1> {
    typedef T const element;
    static constexpr size_t N = 1;
    static constexpr offset_t C = -1;
    typedef get_dtype_t<T const> dtype_t;
    typedef ArrayImpl<T,1> impl_t;
    typedef Layout<1> layout_t;
    typedef typename dtype_t::value_type value_type;
    typedef typename dtype_t::const_reference reference;
    typedef typename dtype_t::const_pointer pointer;
    typedef pointer iterator;

    static reference make_reference_at(
        byte_t * buffer,
        ArrayBase<T const,1,-1> const & self
    ) {
        return *reinterpret_cast<T const*>(buffer);
    }

    static iterator make_iterator_at(
        byte_t * buffer,
        ArrayBase<T const,1,-1> const & parent
    ) {
        return reinterpret_cast<T*>(buffer);
    }
};

template <typename T>
struct ArrayTraits<T const,1,0> {
    typedef T const element;
    static constexpr size_t N = 1;
    static constexpr offset_t C = 1;
    typedef get_dtype_t<T const> dtype_t;
    typedef ArrayImpl<T,1> impl_t;
    typedef Layout<1> layout_t;
    typedef typename dtype_t::value_type value_type;
    typedef typename dtype_t::const_reference reference;
    typedef typename dtype_t::const_pointer pointer;
    typedef Iter<T const,1,0> iterator;

    static reference make_reference_at(
        byte_t * buffer,
        ArrayBase<T const,1,0> const & self
    ) {
        return *reinterpret_cast<T const*>(buffer);
    }

    static iterator make_iterator_at(
        byte_t * buffer,
        ArrayBase<T const,1,0> const &
    );

};

} // namespace detail
} // namespace ndarray

#endif // !NDARRAY_detail_ArrayTraits_1_hpp_INCLUDED
