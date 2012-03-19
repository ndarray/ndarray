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
