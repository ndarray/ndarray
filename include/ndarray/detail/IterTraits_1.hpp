
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

template <typename T>
struct IterTraits {
    typedef byte_t * storage;
    typedef typename std::remove_const<T>::type value_type;
    typedef T & reference;
    typedef T * pointer;
    typedef reference offset_ref;
    typedef std::random_access_iterator_tag category;

    static void reset(storage & s, storage & other) { s = other; }

    static reference dereference(storage s) {
        return *reinterpret_cast<T*>(s);
    }

    static offset_ref dereference_at(storage s, offset_t nbytes) {
        return *reinterpret_cast<T*>(s + nbytes);
    }

    static pointer get_pointer(storage s) {
        return reinterpret_cast<T*>(s);
    }

    static void advance(storage & s, offset_t nbytes) {
        s += nbytes;
    }

    static byte_t * buffer(storage s) {
        return s;
    }

};

template <typename T, size_t N, offset_t C>
struct IterTraits<Array<T,N,C>> {
    typedef ArrayRef<T,N,C> storage;
    typedef Array<T,N,C> value_type;
    typedef ArrayRef<T,N,C> const & reference;
    typedef Array<T,N,C> const * pointer;
    typedef ArrayRef<T,N,C> offset_ref;
    typedef std::input_iterator_tag category;

    template <typename Other>
    static void reset(storage & s, Other const & other);

    static reference dereference(storage const & s);

    static offset_ref dereference_at(storage const & s, offset_t nbytes);

    static pointer get_pointer(storage const & s);

    static void advance(storage & s, offset_t nbytes);

    static byte_t * buffer(storage const & s);

};

} // namespace detail
} // namespace ndarray

#endif // !NDARRAY_detail_IterTraits_1_hpp_INCLUDED
