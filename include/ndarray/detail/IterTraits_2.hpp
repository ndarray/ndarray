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
#include "ndarray/detail/ArrayTraits_2.hpp"
#include "ndarray/detail/IterImpl.hpp"
#include "ndarray/detail/Iter_1.hpp"
#include "ndarray/detail/IterTraits_1.hpp"

namespace ndarray {
namespace detail {

template <typename T, size_t N, offset_t C>
inline typename IterTraits<T,N,C>::offset_ref
IterTraits<T,N,C>::make_reference_at(
    byte_t * buffer,
    Iter<T,N,C> const & self
) {
    return ArrayTraits<T,N,C>::make_reference_at(buffer, self._current);
}

template <typename T, size_t N, offset_t C>
inline typename IterTraits<T,N,C>::actual_ref
IterTraits<T,N,C>::make_reference(
    Iter<T,N,C> const & self
) {
    return self._impl._current;
}

template <typename T, size_t N, offset_t C>
inline typename IterTraits<T,N,C>::actual_ptr
IterTraits<T,N,C>::make_pointer(
    Iter<T,N,C> const & self
) {
    return &self._impl._current;
}

template <typename T>
inline typename IterTraits<T,1,0>::offset_ref
IterTraits<T,1,0>::make_reference_at(
    byte_t * buffer,
    Iter<T,1,0> const & self
) {
    return *reinterpret_cast<T*>(buffer);
}

template <typename T>
inline typename IterTraits<T,1,0>::actual_ref
IterTraits<T,1,0>::make_reference(Iter<T,1,0> const & self) {
    return *reinterpret_cast<T*>(self._impl.buffer());
}

template <typename T>
inline typename IterTraits<T,1,0>::actual_ptr
IterTraits<T,1,0>::make_pointer(Iter<T,1,0> const & self) {
    return reinterpret_cast<T*>(self._impl.buffer());
}

} // namespace detail
} // namespace ndarray

#endif // !NDARRAY_IterTraits_2_hpp_INCLUDED
