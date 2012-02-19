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
#ifndef NDARRAY_FFT_fft_fwd_h_INCLUDED
#define NDARRAY_FFT_fft_fwd_h_INCLUDED

/**
 * @file ndarray/fft_fwd.h 
 *
 * @brief Forward declarations and default template parameters for ndarray/fft.
 *
 *  \note This file is not included by the main "ndarray.h" header file.
 */

/**
 *  \defgroup FFTGroup Fourier Transforms
 *
 *  @brief Fast fourier transforms using the FFTW library.
 */

/// @internal \defgroup FFTndarrayInternalGroup Fourier Transform Internals

#include "ndarray_fwd.h"

namespace ndarray {
namespace detail {

template <typename T, bool IsConst=boost::is_const<T>::value> struct FourierTraits;
template <typename T> struct FFTWTraits;

} // namespace detail

template <typename T, int N> class FourierTransform;

} // namespace ndarray

#endif // !NDARRAY_FFT_fft_fwd_h_INCLUDED
