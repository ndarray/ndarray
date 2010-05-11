#ifndef LSST_NDARRAY_fft_hpp_INCLUDED
#define LSST_NDARRAY_fft_hpp_INCLUDED

/**
 * @file lsst/ndarray/fft.hpp 
 *
 * @brief Main public header file for ndarray FFT library.
 *
 *  \note This file is not included by the main "lsst/ndarray.hpp" header file.
 */

#include "lsst/ndarray.hpp"
#include "lsst/ndarray/fft/FourierTransform.hpp"
#include "lsst/ndarray/fft/FourierOps.hpp"
#ifndef LSST_NDARRAY_FFT_MANUAL_INCLUDE
#include "lsst/ndarray/fft/FourierTransform.cc"
#endif

#endif // !LSST_NDARRAY_fft_hpp_INCLUDED
