
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

template <typename T, int N>
struct IterTraits : public ArrayTraits<T,N> {
    typedef IterImpl<typename std::remove_const<T>::type,N> impl_t;
    typedef std::input_iterator_tag category;
    typedef ArrayRef<T,N-1> const & actual_ref;
    typedef Array<T,N-1> const * actual_ptr;

    using ArrayTraits<T,N>::make_reference_at;

    static ArrayRef<T,N-1> make_reference_at(
        byte_t * buffer,
        Iter<T,N> const & self
    );

    static actual_ref make_reference(Iter<T,N> const & self);

    static actual_ptr make_pointer(Iter<T,N> const & self);

    static Iter<T,N> make_iterator_at(
        byte_t * buffer,
        ArrayBase<T,N> const & parent
    );
};

template <typename T>
struct IterTraits<T,1> : public ArrayTraits<T,1> {
    typedef IterImpl<typename std::remove_const<T>::type,1> impl_t;
    typedef std::random_access_iterator_tag category;
    typedef T & actual_ref;
    typedef T * actual_ptr;

    using ArrayTraits<T,1>::make_reference_at;

    static T & make_reference_at(
        byte_t * buffer,
        Iter<T,1> const & self
    );

    static actual_ref make_reference(Iter<T,1> const & self);

    static actual_ptr make_pointer(Iter<T,1> const & self);

    static Iter<T,1> make_iterator_at(
        byte_t * buffer,
        ArrayBase<T,1> const & parent
    );
};

} // namespace detail
} // namespace ndarray

#endif // !NDARRAY_detail_IterTraits_1_hpp_INCLUDED
