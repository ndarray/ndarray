// -*- c++ -*-
/*
 * Copyright 2012, Jim Bosch and the LSST Corporation
 * 
 * ndarray is available under two licenses, both of which are described
 * more fully in other files that should be distributed along with
 * the code:
 * 
 *  - A simple BSD-style license (ndarray-bsd-license.txt); under this
 *    license ndarray is broadly compatible with essentially any other
 *    code.
 * 
 *  - As a part of the LSST data management software system, ndarray is
 *    licensed with under the GPL v3 (LsstLicenseStatement.txt).
 * 
 * These files can also be found in the source distribution at:
 * 
 * https://github.com/ndarray/ndarray
 */
#ifndef NDARRAY_FFT_FourierTraits_h_INCLUDED
#define NDARRAY_FFT_FourierTraits_h_INCLUDED

/** 
 *  @file ndarray/fft/FourierTraits.h
 *
 *  @brief Traits classes to handle real-data and complex-data FFTs in a template-friendly way.
 */

#include <complex>
#include <boost/shared_ptr.hpp>

#include "ndarray/fft_fwd.h"

namespace ndarray {
/// \cond INTERNAL
namespace detail {

/**
 *  @internal @ingroup FFTndarrayInternalGroup
 *  @brief A traits class that defines x- and k-space data types and k-space array sizes.
 */
template <typename T, bool IsConst>
struct FourierTraits {
    BOOST_STATIC_ASSERT(sizeof(T) < 0);
};

/// \cond SPECIALIZATIONS

template <typename T>
struct FourierTraits<T,false> {
    typedef T ElementX;
    typedef T ValueX;
    typedef std::complex<T> ElementK;
    typedef std::complex<T> ValueK;

    static inline int computeLastDimensionSize(int n) { return n/2 + 1; }
};

template <typename T>
struct FourierTraits<T,true> {
    typedef T ElementX;
    typedef typename boost::remove_const<T>::type ValueX;
    typedef std::complex<ValueX> ValueK;
    typedef ValueK const ElementK;

    static inline int computeLastDimensionSize(int n) { return n/2 + 1; }
};

template <typename U>
struct FourierTraits<std::complex<U>,false> {
    typedef std::complex<U> ElementX;
    typedef std::complex<U> ElementK;
    typedef std::complex<U> ValueX;
    typedef std::complex<U> ValueK;

    static inline int computeLastDimensionSize(int n) { return n; }
};

template <typename U>
struct FourierTraits<std::complex<U> const,true> {
    typedef std::complex<U> const ElementX;
    typedef std::complex<U> const ElementK;
    typedef std::complex<U> ValueX;
    typedef std::complex<U> ValueK;

    static inline int computeLastDimensionSize(int n) { return n; }
};

/// \endcond

} // namespace detail
/// \endcond
} // namespace ndarray

#endif // !NDARRAY_FFT_FourierTraits_h_INCLUDED
