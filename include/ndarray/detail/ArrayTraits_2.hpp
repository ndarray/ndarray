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
#ifndef NDARRAY_ArrayTraits_2_hpp_INCLUDED
#define NDARRAY_ArrayTraits_2_hpp_INCLUDED

#include "ndarray/common.hpp"
#include "ndarray/detail/ArrayImpl.hpp"
#include "ndarray/detail/ArrayBase_1.hpp"
#include "ndarray/detail/ArrayRef_1.hpp"
#include "ndarray/detail/ArrayTraits_1.hpp"
#include "ndarray/detail/IterImpl.hpp"

namespace ndarray {
namespace detail {

template <typename T, size_t N, offset_t C>
typename ArrayTraits<T,N,C>::reference
ArrayTraits<T,N,C>::make_reference_at(
    byte_t * buffer,
    ArrayBase<T,N,C> const & self
) {
    return typename ArrayTraits<T,N,C>::reference(
        typename ArrayTraits<T,N-1,nestedC>::impl_t(
            buffer,
            self._impl.layout(),
            self._impl.manager(),
            self._impl.dtype()
        )
    );
}

template <typename T, size_t N, offset_t C>
inline typename ArrayTraits<T,N,C>::iterator
ArrayTraits<T,N,C>::make_iterator_at(
    byte_t * buffer,
    ArrayBase<T,N,C> const & parent
) {
    return Iter<T,N,C>(
        IterImpl<typename std::remove_const<T>::type,N,C>(
            make_reference_at(buffer, parent),
            parent.stride()
        )
    );
}

template <typename T>
inline typename ArrayTraits<T,1,0>::iterator
ArrayTraits<T,1,0>::make_iterator_at(
    byte_t * buffer,
    ArrayBase<T,1,0> const & parent
) {
    return Iter<T,1,0>(
        IterImpl<T,N,C>(buffer, parent.dtype(), parent.stride())
    );
}

template <typename T>
inline typename ArrayTraits<T const,1,0>::iterator
ArrayTraits<T const,1,0>::make_iterator_at(
    byte_t * buffer,
    ArrayBase<T const,1,0> const & parent
) {
    return Iter<T const,1,0>(
        impl_t(buffer, parent.dtype(), parent.stride())
    );
}

} // namespace detail
} // namespace ndarray

#endif // !NDARRAY_ArrayTraits_2_hpp_INCLUDED
