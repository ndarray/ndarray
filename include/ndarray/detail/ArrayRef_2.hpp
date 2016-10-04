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
#ifndef NDARRAY_detail_ArrayRef_2_hpp_INCLUDED
#define NDARRAY_detail_ArrayRef_2_hpp_INCLUDED

#include "ndarray/common.hpp"
#include "ndarray/detail/ArrayRef_1.hpp"
#include "ndarray/detail/Array_2.hpp"

namespace ndarray {

template <typename T, size_t N, offset_t C>

inline ArrayRef<T,N,C> const &
ArrayRef<T,N,C>::operator=(Array<T const,N,C> const & other) const {
    std::copy(other.begin(), other.end(), this->begin());
    return *this;
}

template <typename T, size_t N, offset_t C>
inline ArrayRef<T,N,C> const &
ArrayRef<T,N,C>::operator=(Array<T,N,C> && other) const {
    std::move(other.begin(), other.end(), this->begin());
    return *this;
}

#ifdef NDARRAY_FAST_CONVERSIONS

    template <typename T, size_t N, offset_t C>
    template <offset_t D>
    inline ArrayRef<T const,N,C>::operator
    ArrayRef<T const,N,D> const & () const {
        static_assert(
            (C >= D && D >= 0) || (C <= D && D <= 0) || (N == 1 && C == -D),
            "Invalid conversion from fewer to more contiguous dimensions."
        );
        return *reinterpret_cast<ArrayRef<T const,N,D> const *>(this);
    }

    template <typename T, size_t N, offset_t C>
    template <offset_t D>
    inline ArrayRef<T,N,C>::operator
    ArrayRef<T,N,D> const & () const {
        static_assert(
            (C >= D && D >= 0) || (C <= D && D <= 0) || (N == 1 && C == -D),
            "Invalid conversion from fewer to more contiguous dimensions."
        );
        return *reinterpret_cast<ArrayRef<T,N,D> const *>(this);
    }

    template <typename T, size_t N, offset_t C>
    template <offset_t D>
    inline ArrayRef<T,N,C>::operator
    ArrayRef<T const,N,D> const & () const {
        static_assert(
            (C >= D && D >= 0) || (C <= D && D <= 0) || (N == 1 && C == -D),
            "Cannot implicitly guarantee more contiguous dimensions."
        );
        return *reinterpret_cast<ArrayRef<T const,N,D> const *>(this);
    }

#else

    template <typename T, size_t N, offset_t C>
    template <offset_t D>
    inline ArrayRef<T const,N,C>::operator ArrayRef<T const,N,D>() const {
        static_assert(
            (C >= D && D >= 0) || (C <= D && D <= 0) || (N == 1 && C == -D),
            "Invalid conversion from fewer to more contiguous dimensions."
        );
        return ArrayRef<T const,N,D>(this->_impl);
    }

    template <typename T, size_t N, offset_t C>
    template <offset_t D>
    inline ArrayRef<T,N,C>::operator ArrayRef<T,N,D>() const {
        static_assert(
            (C >= D && D >= 0) || (C <= D && D <= 0) || (N == 1 && C == -D),
            "Invalid conversion from fewer to more contiguous dimensions."
        );
        return ArrayRef<T,N,D>(this->_impl);
    }

    template <typename T, size_t N, offset_t C>
    template <offset_t D>
    inline ArrayRef<T,N,C>::operator ArrayRef<T const,N,D>() const {
        static_assert(
            (C >= D && D >= 0) || (C <= D && D <= 0) || (N == 1 && C == -D),
            "Invalid conversion from fewer to more contiguous dimensions."
        );
        return ArrayRef<T const,N,D>(this->_impl);
    }

#endif

template <typename T, size_t N, offset_t C>
inline ArrayRef<T,N,C> const & ArrayRef<T,N,C>::operator=(T scalar) const {
    std::fill(this->begin(), this->end(), scalar);
    return *this;
}


} // namespace ndarray

#endif // !NDARRAY_detail_ArrayRef_2_hpp_INCLUDED
