// -*- lsst-c++ -*-
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
#ifndef NDARRAY_FFT_FFTWTraits_h_INCLUDED
#define NDARRAY_FFT_FFTWTraits_h_INCLUDED

/** 
 *  @file ndarray/fft/FFTWTraits.h
 *
 *  \brief Traits classes that wrap FFTW in a template-friendly interface.
 */

#include <complex>
#include <fftw3.h>
#include "ndarray/fft/FourierTraits.h"

namespace ndarray {
/// \cond INTERNAL
namespace detail {

/**
 *  \internal \ingroup FFTInternalGroup
 *  \brief A traits class that maps C++ template types to FFTW types and wraps FFTW function calls.
 */
template <typename T> struct FFTWTraits { BOOST_STATIC_ASSERT(sizeof(T) < 0); };

/// \cond SPECIALIZATIONS

    template <> struct FFTWTraits<float> {
        BOOST_STATIC_ASSERT((!boost::is_const<float>::value));
        typedef fftwf_plan Plan;
        typedef FourierTraits<float>::ElementX ElementX;
        typedef FourierTraits<float>::ElementK ElementK;
        typedef boost::shared_ptr<ElementX> OwnerX;
        typedef boost::shared_ptr<ElementK> OwnerK;
        static inline Plan forward(int rank, const int *n, int howmany,	
                                   ElementX *in, const int *inembed, int istride, int idist,
                                   ElementK *out, const int *onembed, int ostride, int odist,
                                   unsigned flags) {			
            return fftwf_plan_many_dft_r2c(rank, n, howmany,
                                        in, inembed, istride, idist,
                                        reinterpret_cast<fftwf_complex*>(out),
                                        onembed, ostride, odist,
                                        flags);			
        }
        static inline Plan inverse(int rank, const int *n, int howmany,
                                   ElementK *in, const int *inembed, int istride, int idist,
                                   ElementX *out, const int *onembed, int ostride, int odist,
                                   unsigned flags) {			
            return fftwf_plan_many_dft_c2r(rank, n, howmany,
                                        reinterpret_cast<fftwf_complex*>(in),
                                        inembed, istride, idist,
                                        out, onembed, ostride, odist,
                                        flags);			
        }
        static inline void destroy(Plan p) { fftwf_destroy_plan(p); }
        static inline void execute(Plan p) { fftwf_execute(p); }	
        static inline OwnerX allocateX(int n) {
            return OwnerX(
                reinterpret_cast<ElementX*>(
                    fftwf_malloc(sizeof(ElementX)*n)
                ),
                fftwf_free
            );
        }
        static inline OwnerK allocateK(int n) {
            return OwnerK(
                reinterpret_cast<ElementK*>(
                    fftwf_malloc(sizeof(ElementK)*n)
                ),
                fftwf_free
            );
        }
    };
    template <> struct FFTWTraits< std::complex<float> > {
        typedef fftwf_plan Plan;
        typedef FourierTraits< std::complex<float> >::ElementX ElementX;
        typedef FourierTraits< std::complex<float> >::ElementK ElementK;
        typedef boost::shared_ptr<ElementX> OwnerX;
        typedef boost::shared_ptr<ElementK> OwnerK;
        static inline Plan forward(int rank, const int *n, int howmany,	
                                   ElementX *in, const int *inembed, int istride, int idist,
                                   ElementK *out, const int *onembed, int ostride, int odist,
                                   unsigned flags) {			
            return fftwf_plan_many_dft(rank, n, howmany,
                                    reinterpret_cast<fftwf_complex*>(in),
                                    inembed, istride, idist,
                                    reinterpret_cast<fftwf_complex*>(out),
                                    onembed, ostride, odist,
                                    FFTW_FORWARD, flags);
        }
        static inline Plan inverse(int rank, const int *n, int howmany,
                                   ElementK *in, const int *inembed, int istride, int idist,
                                   ElementX *out, const int *onembed, int ostride, int odist,
                                   unsigned flags) {			
            return fftwf_plan_many_dft(rank, n, howmany,
                                    reinterpret_cast<fftwf_complex*>(in),
                                    inembed, istride, idist,
                                    reinterpret_cast<fftwf_complex*>(out),
                                    onembed, ostride, odist,
                                    FFTW_BACKWARD,flags);
        }
        static inline void destroy(Plan p) { fftwf_destroy_plan(p); }
        static inline void execute(Plan p) { fftwf_execute(p); }	
        static inline OwnerX allocateX(int n) {
            return OwnerX(
                reinterpret_cast<ElementX*>(
                    fftwf_malloc(sizeof(ElementX)*n)
                ),
                fftwf_free
            );
        }
        static inline OwnerK allocateK(int n) {
            return OwnerK(
                reinterpret_cast<ElementK*>(
                    fftwf_malloc(sizeof(ElementK)*n)
                ),
                fftwf_free
            );
        }
    };

    template <> struct FFTWTraits<double> {
        BOOST_STATIC_ASSERT((!boost::is_const<double>::value));
        typedef fftw_plan Plan;
        typedef FourierTraits<double>::ElementX ElementX;
        typedef FourierTraits<double>::ElementK ElementK;
        typedef boost::shared_ptr<ElementX> OwnerX;
        typedef boost::shared_ptr<ElementK> OwnerK;
        static inline Plan forward(int rank, const int *n, int howmany,	
                                   ElementX *in, const int *inembed, int istride, int idist,
                                   ElementK *out, const int *onembed, int ostride, int odist,
                                   unsigned flags) {			
            return fftw_plan_many_dft_r2c(rank, n, howmany,
                                        in, inembed, istride, idist,
                                        reinterpret_cast<fftw_complex*>(out),
                                        onembed, ostride, odist,
                                        flags);			
        }
        static inline Plan inverse(int rank, const int *n, int howmany,
                                   ElementK *in, const int *inembed, int istride, int idist,
                                   ElementX *out, const int *onembed, int ostride, int odist,
                                   unsigned flags) {			
            return fftw_plan_many_dft_c2r(rank, n, howmany,
                                        reinterpret_cast<fftw_complex*>(in),
                                        inembed, istride, idist,
                                        out, onembed, ostride, odist,
                                        flags);			
        }
        static inline void destroy(Plan p) { fftw_destroy_plan(p); }
        static inline void execute(Plan p) { fftw_execute(p); }	
        static inline OwnerX allocateX(int n) {
            return OwnerX(
                reinterpret_cast<ElementX*>(
                    fftw_malloc(sizeof(ElementX)*n)
                ),
                fftw_free
            );
        }
        static inline OwnerK allocateK(int n) {
            return OwnerK(
                reinterpret_cast<ElementK*>(
                    fftw_malloc(sizeof(ElementK)*n)
                ),
                fftw_free
            );
        }
    };
    template <> struct FFTWTraits< std::complex<double> > {
        typedef fftw_plan Plan;
        typedef FourierTraits< std::complex<double> >::ElementX ElementX;
        typedef FourierTraits< std::complex<double> >::ElementK ElementK;
        typedef boost::shared_ptr<ElementX> OwnerX;
        typedef boost::shared_ptr<ElementK> OwnerK;
        static inline Plan forward(int rank, const int *n, int howmany,	
                                   ElementX *in, const int *inembed, int istride, int idist,
                                   ElementK *out, const int *onembed, int ostride, int odist,
                                   unsigned flags) {			
            return fftw_plan_many_dft(rank, n, howmany,
                                    reinterpret_cast<fftw_complex*>(in),
                                    inembed, istride, idist,
                                    reinterpret_cast<fftw_complex*>(out),
                                    onembed, ostride, odist,
                                    FFTW_FORWARD, flags);
        }
        static inline Plan inverse(int rank, const int *n, int howmany,
                                   ElementK *in, const int *inembed, int istride, int idist,
                                   ElementX *out, const int *onembed, int ostride, int odist,
                                   unsigned flags) {			
            return fftw_plan_many_dft(rank, n, howmany,
                                    reinterpret_cast<fftw_complex*>(in),
                                    inembed, istride, idist,
                                    reinterpret_cast<fftw_complex*>(out),
                                    onembed, ostride, odist,
                                    FFTW_BACKWARD,flags);
        }
        static inline void destroy(Plan p) { fftw_destroy_plan(p); }
        static inline void execute(Plan p) { fftw_execute(p); }	
        static inline OwnerX allocateX(int n) {
            return OwnerX(
                reinterpret_cast<ElementX*>(
                    fftw_malloc(sizeof(ElementX)*n)
                ),
                fftw_free
            );
        }
        static inline OwnerK allocateK(int n) {
            return OwnerK(
                reinterpret_cast<ElementK*>(
                    fftw_malloc(sizeof(ElementK)*n)
                ),
                fftw_free
            );
        }
    };
#ifndef NDARRAY_FFT_NO_LONG_DOUBLE

    template <> struct FFTWTraits<long double> {
        BOOST_STATIC_ASSERT((!boost::is_const<long double>::value));
        typedef fftwl_plan Plan;
        typedef FourierTraits<long double>::ElementX ElementX;
        typedef FourierTraits<long double>::ElementK ElementK;
        typedef boost::shared_ptr<ElementX> OwnerX;
        typedef boost::shared_ptr<ElementK> OwnerK;
        static inline Plan forward(int rank, const int *n, int howmany,	
                                   ElementX *in, const int *inembed, int istride, int idist,
                                   ElementK *out, const int *onembed, int ostride, int odist,
                                   unsigned flags) {			
            return fftwl_plan_many_dft_r2c(rank, n, howmany,
                                        in, inembed, istride, idist,
                                        reinterpret_cast<fftwl_complex*>(out),
                                        onembed, ostride, odist,
                                        flags);			
        }
        static inline Plan inverse(int rank, const int *n, int howmany,
                                   ElementK *in, const int *inembed, int istride, int idist,
                                   ElementX *out, const int *onembed, int ostride, int odist,
                                   unsigned flags) {			
            return fftwl_plan_many_dft_c2r(rank, n, howmany,
                                        reinterpret_cast<fftwl_complex*>(in),
                                        inembed, istride, idist,
                                        out, onembed, ostride, odist,
                                        flags);			
        }
        static inline void destroy(Plan p) { fftwl_destroy_plan(p); }
        static inline void execute(Plan p) { fftwl_execute(p); }	
        static inline OwnerX allocateX(int n) {
            return OwnerX(
                reinterpret_cast<ElementX*>(
                    fftwl_malloc(sizeof(ElementX)*n)
                ),
                fftwl_free
            );
        }
        static inline OwnerK allocateK(int n) {
            return OwnerK(
                reinterpret_cast<ElementK*>(
                    fftwl_malloc(sizeof(ElementK)*n)
                ),
                fftwl_free
            );
        }
    };
    template <> struct FFTWTraits< std::complex<long double> > {
        typedef fftwl_plan Plan;
        typedef FourierTraits< std::complex<long double> >::ElementX ElementX;
        typedef FourierTraits< std::complex<long double> >::ElementK ElementK;
        typedef boost::shared_ptr<ElementX> OwnerX;
        typedef boost::shared_ptr<ElementK> OwnerK;
        static inline Plan forward(int rank, const int *n, int howmany,	
                                   ElementX *in, const int *inembed, int istride, int idist,
                                   ElementK *out, const int *onembed, int ostride, int odist,
                                   unsigned flags) {			
            return fftwl_plan_many_dft(rank, n, howmany,
                                    reinterpret_cast<fftwl_complex*>(in),
                                    inembed, istride, idist,
                                    reinterpret_cast<fftwl_complex*>(out),
                                    onembed, ostride, odist,
                                    FFTW_FORWARD, flags);
        }
        static inline Plan inverse(int rank, const int *n, int howmany,
                                   ElementK *in, const int *inembed, int istride, int idist,
                                   ElementX *out, const int *onembed, int ostride, int odist,
                                   unsigned flags) {			
            return fftwl_plan_many_dft(rank, n, howmany,
                                    reinterpret_cast<fftwl_complex*>(in),
                                    inembed, istride, idist,
                                    reinterpret_cast<fftwl_complex*>(out),
                                    onembed, ostride, odist,
                                    FFTW_BACKWARD,flags);
        }
        static inline void destroy(Plan p) { fftwl_destroy_plan(p); }
        static inline void execute(Plan p) { fftwl_execute(p); }	
        static inline OwnerX allocateX(int n) {
            return OwnerX(
                reinterpret_cast<ElementX*>(
                    fftwl_malloc(sizeof(ElementX)*n)
                ),
                fftwl_free
            );
        }
        static inline OwnerK allocateK(int n) {
            return OwnerK(
                reinterpret_cast<ElementK*>(
                    fftwl_malloc(sizeof(ElementK)*n)
                ),
                fftwl_free
            );
        }
    };
#endif
/// \endcond

} // namespace detail
/// \endcond
} // namespace ndarray

#endif // !NDARRAY_FFT_FFTWTraits_h_INCLUDED
