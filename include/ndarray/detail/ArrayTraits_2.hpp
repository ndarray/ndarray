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

namespace ndarray {
namespace detail {

template <typename T, size_t N>
typename ArrayTraits<T,N>::reference
ArrayTraits<T,N>::make_reference_at(
    byte_t * buffer,
    ArrayBase<T,N> const & self
) {
    return ArrayRef<T,N-1>(
        typename ArrayTraits<T,N-1>::impl_t(
            buffer,
            self._impl.layout(),
            self._impl.manager(),
            self._impl.dtype()
        )
    );
}

} // namespace detail
} // namespace ndarray

#endif // !NDARRAY_ArrayTraits_2_hpp_INCLUDED
