include(`FFTWTraits.macros.m4')dnl
changecom(`###')dnl
#ifndef LSST_NDARRAY_FFT_FFTWTraits_hpp_INCLUDED
#define LSST_NDARRAY_FFT_FFTWTraits_hpp_INCLUDED

/** 
 *  @file lsst/ndarray/fft/FFTWTraits.hpp
 *
 *  \brief Traits classes that wrap FFTW in a template-friendly interface.
 */

#include <complex>
#include <fftw3.h>
#include "lsst/ndarray/fft/FourierTraits.hpp"

namespace lsst { namespace ndarray {
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
#ifndef LSST_NDARRAY_NO_LONG_DOUBLE
FFTW_TRAITS(long double, fftwl);
#endif
/// \endcond

} // namespace lsst:ndarray::detail
/// \endcond
}} // namespace lsst::ndarray

#endif // !LSST_NDARRAY_FFT_FFTWTraits_hpp_INCLUDED
