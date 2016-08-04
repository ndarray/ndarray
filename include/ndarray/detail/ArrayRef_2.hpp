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
#ifndef NDARRAY_ArrayRef_2_hpp_INCLUDED
#define NDARRAY_ArrayRef_2_hpp_INCLUDED

#include "ndarray/common.hpp"
#include "ndarray/detail/ArrayRef_1.hpp"
#include "ndarray/detail/Iter_2.hpp"

namespace ndarray {

template <typename T, size_t N>
inline ArrayRef<T,N> const &
ArrayRef<T,N>::operator=(Array<T const,N> const & other) const {
    std::copy(other.begin(), other.end(), this->begin());
    return *this;
}

template <typename T, size_t N>
inline ArrayRef<T,N> const &
ArrayRef<T,N>::operator=(Array<T,N> && other) const {
    std::move(other.begin(), other.end(), this->begin());
    return *this;
}

#ifndef NDARRAY_FAST_CONVERSIONS

    template <typename T, size_t N>
    inline ArrayRef<T,N> const &
    ArrayRef<T,N>::operator=(Array<T,N> const & other) const {
        std::copy(other.begin(), other.end(), this->begin());
        return *this;
    }

#endif

} // namespace ndarray

