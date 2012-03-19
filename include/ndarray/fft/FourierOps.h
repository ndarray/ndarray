// -*- c++ -*-
/* 
 * LSST Data Management System
 * Copyright 2008, 2009, 2010, 2011 LSST Corporation.
 * 
 * This product includes software developed by the
 * LSST Project (http://www.lsst.org/).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the LSST License Statement and 
 * the GNU General Public License along with this program.  If not, 
 * see <http://www.lsstcorp.org/LegalNotices/>.
 */
#ifndef NDARRAY_FFT_FourierOps_h_INCLUDED
#define NDARRAY_FFT_FourierOps_h_INCLUDED

/** 
 *  @file ndarray/fft/FourierOps.h
 *
 *  @brief Common Fourier-space operations.
 */

#include <boost/noncopyable.hpp>

#include "ndarray.h"

namespace ndarray {
/// \cond INTERNAL
namespace detail {

/**
 *  @internal @ingroup FFTndarrayInternalGroup
 *  @brief Implementations for the shift() and differentiate() functions.
 */
template <typename T, int N>
struct FourierOps {

    template <int C>
    static void shift(
        T const * offset,
        std::complex<T> const & factor,
        ArrayRef<std::complex<T>,N,C> const & array,
        int const real_last_dim
    ) {
        typename ArrayRef<std::complex<T>,N,C>::Iterator iter = array.begin();
        T u = -2.0 * M_PI * (*offset) / array.size();
        int kMid = (array.size() + 1) / 2;
        for (int k = 0; k < kMid; ++k, ++iter) {
            FourierOps<T,N-1>::shift(offset+1, factor * std::polar(static_cast<T>(1), u * k), 
                                     *iter, real_last_dim);
        }
        if (array.size() % 2 == 0) {
            FourierOps<T,N-1>::shift(offset+1, factor * std::cos(u * kMid), *iter, real_last_dim);
            ++iter;
            ++kMid;
        }
        for (int k_n = kMid - array.size(); k_n < 0; ++k_n, ++iter) {
            FourierOps<T,N-1>::shift(offset+1, factor * std::polar(static_cast<T>(1), u * k_n),
                                     *iter, real_last_dim);
        }
    }

    template <int C>
    static void differentiate(int m, ArrayRef<std::complex<T>,N,C> const & array, int const real_last_dim) {
        typename ArrayRef<std::complex<T>,N,C>::Iterator iter = array.begin();
        int kMid = (array.size() + 1) / 2;
        T u = 2.0 * M_PI / array.size();
        for (int k = 0; k < kMid; ++k, ++iter) {
            if (m == N) (*iter) *= std::complex<T>(static_cast<T>(0), u * T(k));
            FourierOps<T,N-1>::differentiate(m, *iter, real_last_dim);
        }
        if (array.size() % 2 == 0) {
            (*iter) = static_cast<T>(0);
            ++iter;
            ++kMid;
        }
        for (int k_n = kMid - array.size(); k_n < 0; ++k_n, ++iter) {
            if (m == N) (*iter) *= std::complex<T>(static_cast<T>(0), u * T(k_n));
            FourierOps<T,N-1>::differentiate(m, *iter, real_last_dim);
        }
    }

};

/**
 *  @internal @ingroup FFTndarrayInternalGroup
 *  @brief Implementations for the shift() and differentiate() functions (1d specialization).
 */
template <typename T>
struct FourierOps<T,1> {
    
    template <int C>
    static void shift(
        T const * offset,
        std::complex<T> const & factor,
        ArrayRef<std::complex<T>,1,C> const & array,
        int const real_last_dim
    ) {
        typename ArrayRef<std::complex<T>,1,C>::Iterator iter = array.begin();
        T u = -2.0 * M_PI * (*offset) / real_last_dim;
        int kMid = (real_last_dim + 1) / 2;
        for (int k = 0; k < kMid; ++k, ++iter) {
            (*iter) *= factor * std::polar(1.0, u * T(k));
        }
        if (real_last_dim % 2 == 0) {
            (*iter) *= factor * std::cos(u * kMid);
            ++iter;
        }
    }

    template <int C>
    static void differentiate(int m, ArrayRef<std::complex<T>,1,C> const & array, int const real_last_dim) {
        typename ArrayRef<std::complex<T>,1,C>::Iterator iter = array.begin();
        int kMid = (real_last_dim + 1) / 2;
        if (m == 1) {
            T u = 2.0 * M_PI / real_last_dim;
            for (int k = 0; k < kMid; ++k, ++iter) {
                (*iter) *= std::complex<T>(static_cast<T>(0), u * T(k));
            }
        }
        if (real_last_dim % 2 == 0) {
            array[kMid] = static_cast<T>(0);
        }            
    }

};

} // namespace detail
/// \endcond

/**
 *  @brief Perform a Fourier-space translation transform.
 *
 *  @ingroup FFTGroup
 */
template <typename T, int N, int C>
void shift(
    Vector<T,N> const & offset,
    Array<std::complex<T>,N,C> const & array,
    int const real_last_dim
) {
    detail::FourierOps<T,N>::shift(
        offset.begin(),
        static_cast< std::complex<T> >(1),
        array.deep(),
        real_last_dim
    );
}

/**
 *  @brief Numerically differentiate the array in Fourier-space in the given dimension.
 *
 *  @ingroup FFTGroup
 */
template <typename T, int N, int C>
void differentiate(
    int n,
    Array<std::complex<T>,N,C> const & array,
    int const real_last_dim
) {
    detail::FourierOps<T,N>::differentiate(N-n, array.deep(), real_last_dim);
}

} // namespace ndarray

#endif // !NDARRAY_FFT_FourierOps_h_INCLUDED
