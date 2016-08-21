
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
#ifndef NDARRAY_detail_IterTraits_1_hpp_INCLUDED
#define NDARRAY_detail_IterTraits_1_hpp_INCLUDED

#include "ndarray/common.hpp"
#include "ndarray/DType.hpp"
#include "ndarray/detail/ArrayTraits_1.hpp"

namespace ndarray {
namespace detail {

template <typename T, size_t N, offset_t C>
struct IterTraits : public ArrayTraits<T,N,C> {
    typedef IterImpl<typename std::remove_const<T>::type,N,C> impl_t;
    typedef std::input_iterator_tag category;
    using ArrayTraits<T,N,C>::nestedC;
    typedef typename ArrayTraits<T,N,C>::reference offset_ref;
    typedef ArrayRef<T,N-1,nestedC> const & actual_ref;
    typedef Array<T,N-1,nestedC> const * actual_ptr;

    using ArrayTraits<T,N,C>::make_reference_at;

    static offset_ref make_reference_at(
        byte_t * buffer,
        Iter<T,N,C> const & self
    );

    static actual_ref make_reference(Iter<T,N,C> const & self);

    static actual_ptr make_pointer(Iter<T,N,C> const & self);

};

template <typename T>
struct IterTraits<T,1,0> : public ArrayTraits<T,1,0> {
    typedef IterImpl<typename std::remove_const<T>::type,1,0> impl_t;
    typedef std::random_access_iterator_tag category;
    typedef T & offset_ref;
    typedef T & actual_ref;
    typedef T * actual_ptr;

    using ArrayTraits<T,1,0>::make_reference_at;

    static offset_ref make_reference_at(
        byte_t * buffer,
        Iter<T,1,0> const & self
    );

    static actual_ref make_reference(Iter<T,1,0> const & self);

    static actual_ptr make_pointer(Iter<T,1,0> const & self);

};

} // namespace detail
} // namespace ndarray

#endif // !NDARRAY_detail_IterTraits_1_hpp_INCLUDED
