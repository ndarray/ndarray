#ifndef NDARRAY_FFT_fft_fwd_hpp_INCLUDED
#define NDARRAY_FFT_fft_fwd_hpp_INCLUDED

/**
 * @file ndarray/fft_fwd.hpp 
 *
 * @brief Forward declarations and default template parameters for ndarray/fft.
 *
 *  \note This file is not included by the main "ndarray.hpp" header file.
 */

/**
 *  \defgroup FFTGroup Fourier Transforms
 *
 *  @brief Fast fourier transforms using the FFTW library.
 */

/// @internal \defgroup FFTInternalGroup Fourier Transform Internals

#include "ndarray_fwd.hpp"

namespace ndarray {
namespace detail {

template <typename T, bool IsConst=boost::is_const<T>::value> struct FourierTraits;
template <typename T> struct FFTWTraits;

} // namespace ndarray::detail

template <typename T, int N> class FourierTransform;

} // namespace ndarray

#endif // !NDARRAY_FFT_fft_fwd_hpp_INCLUDED
