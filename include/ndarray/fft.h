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
#ifndef NDARRAY_fft_h_INCLUDED
#define NDARRAY_fft_h_INCLUDED

/**
 * @file ndarray/fft.h 
 *
 * @brief Main public header file for ndarray FFT library.
 *
 *  \note This file is not included by the main "ndarray.h" header file.
 */

#include "ndarray.h"
#include "ndarray/fft/FourierTransform.h"
#include "ndarray/fft/FourierOps.h"
#ifndef NDARRAY_FFT_MANUAL_INCLUDE
#include "ndarray/fft/FourierTransform.cc"
#endif

#endif // !NDARRAY_fft_h_INCLUDED
