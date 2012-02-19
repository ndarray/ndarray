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
#include "ndarray/fft/FFTWTraits.h"
#include "ndarray/fft/FourierTransform.h"

namespace ndarray {

template <typename T, int N> 
template <int M>
Array<typename FourierTransform<T,N>::ElementX,M,M>
FourierTransform<T,N>::initializeX(Vector<int,M> const & shape) {
    OwnerX xOwner = detail::FFTWTraits<T>::allocateX(shape.product());
    return Array<ElementX,M,M>(external(xOwner.get(), shape, ROW_MAJOR, xOwner));
}

template <typename T, int N> 
template <int M>
Array<typename FourierTransform<T,N>::ElementK,M,M>
FourierTransform<T,N>::initializeK(Vector<int,M> const & shape) {
    Vector<int,M> kShape(shape);
    kShape[M-1] = detail::FourierTraits<T>::computeLastDimensionSize(shape[M-1]);
    OwnerK kOwner = detail::FFTWTraits<T>::allocateK(kShape.product());
    return Array<ElementK,M,M>(external(kOwner.get(), kShape, ROW_MAJOR, kOwner));
}

template <typename T, int N> 
template <int M>
void
FourierTransform<T,N>::initialize(
    Vector<int,M> const & shape, 
    Array<ElementX,M,M> & x,
    Array<ElementK,M,M> & k
) {
    if (x.empty()) x = initializeX(shape);
    if (k.empty()) k = initializeK(shape);
    NDARRAY_ASSERT(x.getShape() == shape);
    NDARRAY_ASSERT(std::equal(shape.begin(), shape.end()-1, k.getShape().begin()));
}

template <typename T, int N> 
typename FourierTransform<T,N>::Ptr
FourierTransform<T,N>::planForward(
    Index const & shape, 
    typename FourierTransform<T,N>::ArrayX & x,
    typename FourierTransform<T,N>::ArrayK & k
) {
    initialize(shape,x,k);
    return Ptr(
        new FourierTransform(
            detail::FFTWTraits<T>::forward(
                N, shape.begin(), 1,
                x.getData(), NULL, 1, 0,
                k.getData(), NULL, 1, 0,
                FFTW_MEASURE | FFTW_DESTROY_INPUT
            ),
            x.getManager(),
            k.getManager()
        )
    );
}

template <typename T, int N>
typename FourierTransform<T,N>::Ptr
FourierTransform<T,N>::planInverse(
    Index const & shape,
    typename FourierTransform<T,N>::ArrayK & k,
    typename FourierTransform<T,N>::ArrayX & x
) {
    initialize(shape,x,k);
    return Ptr(
        new FourierTransform(
            detail::FFTWTraits<T>::inverse(
                N, shape.begin(), 1,
                k.getData(), NULL, 1, 0,
                x.getData(), NULL, 1, 0,
                FFTW_MEASURE | FFTW_DESTROY_INPUT
            ),
            x.getManager(),
            k.getManager()
        )
    );
}

template <typename T, int N>
typename FourierTransform<T,N>::Ptr
FourierTransform<T,N>::planMultiplexForward(
    MultiplexIndex const & shape,
    typename FourierTransform<T,N>::MultiplexArrayX & x,
    typename FourierTransform<T,N>::MultiplexArrayK & k
) {
    initialize(shape,x,k);
    return Ptr(
        new FourierTransform(
            detail::FFTWTraits<T>::forward(
                N, shape.begin()+1, shape[0],
                x.getData(), NULL, 1, x.template getStride<0>(),
                k.getData(), NULL, 1, k.template getStride<0>(),
                FFTW_MEASURE | FFTW_DESTROY_INPUT
            ),
            x.getManager(),
            k.getManager()
        )
    );
}

template <typename T, int N> 
typename FourierTransform<T,N>::Ptr
FourierTransform<T,N>::planMultiplexInverse(
    MultiplexIndex const & shape,
    typename FourierTransform<T,N>::MultiplexArrayK & k,
    typename FourierTransform<T,N>::MultiplexArrayX & x
) {
    initialize(shape,x,k);
    return Ptr(
        new FourierTransform(
            detail::FFTWTraits<T>::inverse(
                N, shape.begin()+1, shape[0],
                k.getData(), NULL, 1, k.template getStride<0>(),
                x.getData(), NULL, 1, x.template getStride<0>(),
                FFTW_MEASURE | FFTW_DESTROY_INPUT
            ),
            x.getManager(),
            k.getManager()
        )
    );
}

template <typename T, int N>
void FourierTransform<T,N>::execute() {
    detail::FFTWTraits<T>::execute(
        reinterpret_cast<typename detail::FFTWTraits<T>::Plan>(_plan)
    );
}

template <typename T, int N>
FourierTransform<T,N>::~FourierTransform() {
    detail::FFTWTraits<T>::destroy(
        reinterpret_cast<typename detail::FFTWTraits<T>::Plan>(_plan)
    );
}

} // namespace ndarray
