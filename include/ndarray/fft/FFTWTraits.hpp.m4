include(`FFTWTraits.macros.m4')dnl
changecom(`###')dnl
#ifndef NDARRAY_FFT_FFTWTraits_hpp_INCLUDED
#define NDARRAY_FFT_FFTWTraits_hpp_INCLUDED

/** 
 *  @file ndarray/fft/FFTWTraits.hpp
 *
 *  \brief Traits classes that wrap FFTW in a template-friendly interface.
 */

#include <complex>
#include <fftw3.h>
#include "ndarray/fft/FourierTraits.hpp"

namespace ndarray {
/// \cond INTERNAL
namespace detail {

/**
 *  \internal \ingroup FFTInternalGroup
 *  \brief A traits class that maps C++ template types to FFTW types and wraps FFTW function calls.
 */
template <typename T> struct FFTWTraits { BOOST_STATIC_ASSERT(sizeof(T) < 0); };

/// \cond SPECIALIZATIONS
FFTW_TRAITS(float,fftwf);
FFTW_TRAITS(double,fftw);
#ifndef NDARRAY_NO_LONG_DOUBLE
FFTW_TRAITS(long double, fftwl);
#endif
/// \endcond

} // namespace ndarray::detail
/// \endcond
} // namespace ndarray

#endif // !NDARRAY_FFT_FFTWTraits_hpp_INCLUDED
