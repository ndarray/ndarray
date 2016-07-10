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

template <typename T, int N>
inline ArrayRef<T,N-1> IterTraits<T,N>::make_reference_at(
    byte_t * buffer,
    Iter<T,N> const & self
) {
    return ArrayTraits<T,N>::make_reference_at(buffer, self._current);
}

template <typename T, int N>
inline ArrayRef<T,N-1> const & IterTraits<T,N>::make_reference(
    Iter<T,N> const & self
) {
    return self._impl._current;
}

template <typename T, int N>
inline Array<T,N-1> const * IterTraits<T,N>::make_pointer(
    Iter<T,N> const & self
) {
    return &self._impl._current;
}

template <typename T, int N>
inline Iter<T,N> IterTraits<T,N>::make_iterator_at(
    byte_t * buffer,
    ArrayBase<T,N> const & parent
) {
    return Iter<T,N>(
        impl_t(make_reference_at(buffer, parent), parent.stride())
    );
}

template <typename T>
inline T & IterTraits<T,1>::make_reference_at(
    byte_t * buffer,
    Iter<T,1> const & self
) {
    return *reinterpret_cast<T*>(buffer);
}

template <typename T>
inline T & IterTraits<T,1>::make_reference(Iter<T,1> const & self) {
    return *reinterpret_cast<T*>(self._impl.buffer());
}

template <typename T>
inline T * IterTraits<T,1>::make_pointer(Iter<T,1> const & self) {
    return reinterpret_cast<T*>(self._impl.buffer());
}

template <typename T>
inline Iter<T,1> IterTraits<T,1>::make_iterator_at(
    byte_t * buffer,
    ArrayBase<T,1> const & parent
) {
    return Iter<T,1>(
        impl_t(buffer, parent.dtype(), parent.stride())
    );
}

} // namespace detail
} // namespace ndarray

#endif // !NDARRAY_IterTraits_2_hpp_INCLUDED
