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
#ifndef NDARRAY_Array_2_hpp_INCLUDED
#define NDARRAY_Array_2_hpp_INCLUDED

#include "ndarray/common.hpp"
#include "ndarray/detail/Array_1.hpp"
#include "ndarray/detail/ArrayRef_1.hpp"
#include "ndarray/detail/ArrayBase_2.hpp"

namespace ndarray {

#ifdef NDARRAY_FAST_CONVERSIONS

    template <typename T, int N>
    inline ArrayRef<T const,N> const & Array<T const,N>::operator*() const {
        return *reinterpret_cast<ArrayRef<T const,N> const *>(this);
    }

    template <typename T, int N>
    inline ArrayRef<T,N> const & Array<T,N>::operator*() const {
        return *reinterpret_cast<ArrayRef<T,N> const *>(this);
    }

    template <typename T, int N>
    inline Array<T,N>::operator Array<T const,N> const & () const {
        return *reinterpret_cast<Array<T const,N> const *>(this);
    }

#else

    template <typename T, int N>
    inline ArrayRef<T const,N> Array<T const,N>::operator*() const {
        return ArrayRef<T const,N>(this->_impl);
    }

    template <typename T, int N>
    inline ArrayRef<T,N> Array<T,N>::operator*() const{
        return ArrayRef<T,N>(this->_impl);
    }

    template <typename T, int N>
    inline Array<T,N>::operator Array<T const,N>() const {
        return Array<T const,N>(this->_impl);
    }

#endif // NDARRAY_FAST_CONVERSIONS

} // namespace ndarray

#endif // !NDARRAY_Array_2_hpp_INCLUDED
