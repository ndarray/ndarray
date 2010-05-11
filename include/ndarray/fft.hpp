#ifndef NDARRAY_fft_hpp_INCLUDED
#define NDARRAY_fft_hpp_INCLUDED

/**
 * @file ndarray/fft.hpp 
 *
 * @brief Main public header file for ndarray FFT library.
 *
 *  \note This file is not included by the main "ndarray.hpp" header file.
 */

#include "ndarray.hpp"
#include "ndarray/fft/FourierTransform.hpp"
#include "ndarray/fft/FourierOps.hpp"
#ifndef NDARRAY_FFT_MANUAL_INCLUDE
#include "ndarray/fft/FourierTransform.cc"
#endif

#endif // !NDARRAY_fft_hpp_INCLUDED
