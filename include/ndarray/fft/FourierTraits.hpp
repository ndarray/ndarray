#ifndef NDARRAY_FFT_FourierTraits_hpp_INCLUDED
#define NDARRAY_FFT_FourierTraits_hpp_INCLUDED

/** 
 *  @file ndarray/fft/FourierTraits.hpp
 *
 *  @brief Traits classes to handle real-data and complex-data FFTs in a template-friendly way.
 */

#include <complex>
#include <boost/shared_ptr.hpp>

#include "ndarray/fft_fwd.hpp"

namespace ndarray {
/// \cond INTERNAL
namespace detail {

/**
 *  @internal @ingroup FFTInternalGroup
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
    typedef std::complex<T> ElementK;

    static inline int computeLastDimensionSize(int n) { return n/2 + 1; }
};

template <typename T>
struct FourierTraits<T,true> {
    typedef T ElementX;
    typedef std::complex<typename boost::remove_const<T>::type> const ElementK;

    static inline int computeLastDimensionSize(int n) { return n/2 + 1; }
};

template <typename U>
struct FourierTraits<std::complex<U>,false> {
    typedef std::complex<U> ElementX;
    typedef std::complex<U> ElementK;

    static inline int computeLastDimensionSize(int n) { return n; }
};

template <typename U>
struct FourierTraits<std::complex<U> const,true> {
    typedef std::complex<U> const ElementX;
    typedef std::complex<U> const ElementK;

    static inline int computeLastDimensionSize(int n) { return n; }
};

/// \endcond

} // namespace ndarray::detail
/// \endcond
} // namespace ndarray

#endif // !NDARRAY_FFT_FourierTraits_hpp_INCLUDED
