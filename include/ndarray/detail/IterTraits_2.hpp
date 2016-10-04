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
#ifndef NDARRAY_IterTraits_2_hpp_INCLUDED
#define NDARRAY_IterTraits_2_hpp_INCLUDED

#include "ndarray/common.hpp"
#include "ndarray/detail/ArrayBase_1.hpp"
#include "ndarray/detail/Array_1.hpp"
#include "ndarray/detail/ArrayRef_1.hpp"
#include "ndarray/detail/IterTraits_1.hpp"

namespace ndarray {
namespace detail {

template <typename T, size_t N, offset_t C>
template <typename Other>
inline void IterTraits<Array<T,N,C>>::reset(storage & s, Other const & other) {
    s.shallow() = other;
}

template <typename T, size_t N, offset_t C>
inline auto IterTraits<Array<T,N,C>>::dereference(
    storage const & s
) -> actual_ref {
    return s;
}

template <typename T, size_t N, offset_t C>
inline auto IterTraits<Array<T,N,C>>::get_pointer(
    storage const & s
) ->actual_ptr {
    return &s;
}

template <typename T, size_t N, offset_t C>
inline void IterTraits<Array<T,N,C>>::advance(storage & s, offset_t nbytes) {
    s._impl.buffer += nbytes;
}

template <typename T, size_t N, offset_t C>
inline byte_t * IterTraits<Array<T,N,C>>::buffer(storage const & s) {
    return s._impl.buffer;
}


} // namespace detail
} // namespace ndarray

#endif // !NDARRAY_IterTraits_2_hpp_INCLUDED
