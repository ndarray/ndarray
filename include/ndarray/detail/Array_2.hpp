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

    template <typename T, size_t N, offset_t C>
    inline ArrayRef<T const,N,C> const & Array<T const,N,C>::operator*() const {
        return *reinterpret_cast<ArrayRef<T const,N,C> const *>(this);
    }

    template <typename T, size_t N, offset_t C>
    template <offset_t D>
    inline Array<T const,N,C>::operator Array<T const,N,D> const & () const {
        static_assert(
            (C >= D && D >= 0) || (C <= D && D <= 0) || (N == 1 && C == -D),
            "Invalid conversion from fewer to more contiguous dimensions."
        );
        return *reinterpret_cast<Array<T const,N,D> const *>(this);
    }

    template <typename T, size_t N, offset_t C>
    inline ArrayRef<T,N,C> const & Array<T,N,C>::operator*() const {
        return *reinterpret_cast<ArrayRef<T,N,C> const *>(this);
    }

    template <typename T, size_t N, offset_t C>
    template <offset_t D>
    inline Array<T,N,C>::operator Array<T,N,D> const & () const {
        static_assert(
            (C >= D && D >= 0) || (C <= D && D <= 0) || (N == 1 && C == -D),
            "Invalid conversion from fewer to more contiguous dimensions."
        );
        return *reinterpret_cast<Array<T,N,D> const *>(this);
    }

    template <typename T, size_t N, offset_t C>
    template <offset_t D>
    inline Array<T,N,C>::operator Array<T const,N,D> const & () const {
        static_assert(
            (C >= D && D >= 0) || (C <= D && D <= 0) || (N == 1 && C == -D),
            "Cannot implicitly guarantee more contiguous dimensions."
        );
        return *reinterpret_cast<Array<T const,N,D> const *>(this);
    }

#else

    template <typename T, size_t N, offset_t C>
    inline ArrayRef<T const,N,C> Array<T const,N,C>::operator*() const {
        return ArrayRef<T const,N,C>(this->_impl);
    }

    template <typename T, size_t N, offset_t C>
    template <offset_t D>
    inline Array<T const,N,C>::operator Array<T const,N,D>() const {
        static_assert(
            (C >= D && D >= 0) || (C <= D && D <= 0) || (N == 1 && C == -D),
            "Invalid conversion from fewer to more contiguous dimensions."
        );
        return Array<T const,N,D>(this->_impl);
    }

    template <typename T, size_t N, offset_t C>
    inline ArrayRef<T,N,C> Array<T,N,C>::operator*() const{
        return ArrayRef<T,N,C>(this->_impl);
    }

    template <typename T, size_t N, offset_t C>
    template <offset_t D>
    inline Array<T,N,C>::operator Array<T,N,D>() const {
        static_assert(
            (C >= D && D >= 0) || (C <= D && D <= 0) || (N == 1 && C == -D),
            "Invalid conversion from fewer to more contiguous dimensions."
        );
        return Array<T,N,D>(this->_impl);
    }

    template <typename T, size_t N, offset_t C>
    template <offset_t D>
    inline Array<T,N,C>::operator Array<T const,N,D>() const {
        static_assert(
            (C >= D && D >= 0) || (C <= D && D <= 0) || (N == 1 && C == -D),
            "Invalid conversion from fewer to more contiguous dimensions."
        );
        return Array<T const,N,D>(this->_impl);
    }

#endif // NDARRAY_FAST_CONVERSIONS

} // namespace ndarray

#endif // !NDARRAY_Array_2_hpp_INCLUDED
