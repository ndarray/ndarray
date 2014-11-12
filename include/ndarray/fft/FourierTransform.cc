// -*- c++ -*-
/*
 * Copyright (c) 2010-2012, Jim Bosch
 * All rights reserved.
 *
 * ndarray is distributed under a simple BSD-like license;
 * see the LICENSE file that should be present in the root
 * of the source distribution, or alternately available at:
 * https://github.com/ndarray/ndarray
 */
#include <cstddef>
#include <exception>
#include <numeric>
#include "ndarray/fft/FFTWTraits.h"
#include "ndarray/fft/FourierTransform.h"

namespace ndarray {

template <typename T, int N>
template <int M>
Array<typename FourierTransform<T,N>::ElementX,M,M>
FourierTransform<T,N>::initializeX(Vector<std::size_t,M> const & shape) {
    if (shape.product() > std::numeric_limits<int>::max()) {
      // fftw3 only supports int many elements.
      throw std::exception();
    }
    OwnerX xOwner = detail::FFTWTraits<T>::allocateX(static_cast<int>(shape.product()));
    if (xOwner.get() == nullptr) throw std::exception();
    return Array<ElementX,M,M>(external(xOwner.get(), shape, ROW_MAJOR, xOwner));
}

template <typename T, int N>
template <int M>
Array<typename FourierTransform<T,N>::ElementK,M,M>
FourierTransform<T,N>::initializeK(Vector<std::size_t,M> const & shape) {
    Vector<std::size_t,M> kShape(shape);
    kShape[M-1] = detail::FourierTraits<T>::computeLastDimensionSize(shape[M-1]);
    if (kShape.product() > std::numeric_limits<int>::max()) {
      // fftw3 only supports int many elements.
      throw std::exception();
    }
    OwnerK kOwner = detail::FFTWTraits<T>::allocateK(static_cast<int>(kShape.product()));
    if (kOwner.get() == nullptr) throw std::exception();
    return Array<ElementK,M,M>(external(kOwner.get(), kShape, ROW_MAJOR, kOwner));
}

template <typename T, int N>
template <int M>
void
FourierTransform<T,N>::initialize(
    Vector<std::size_t,M> const & shape,
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
    auto ftshape = ndarray::Vector<int, N>(shape);
    void *fp = detail::FFTWTraits<T>::forward(
                N, ftshape.begin(), 1,
                x.getData(), NULL, 1, 0,
                k.getData(), NULL, 1, 0,
                FFTW_MEASURE | FFTW_DESTROY_INPUT);
    if (fp == nullptr)
      throw std::exception();
    return Ptr(
        new FourierTransform(
            fp,
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
    auto ftshape = ndarray::Vector<int, N>(shape);
    return Ptr(
        new FourierTransform(
            detail::FFTWTraits<T>::inverse(
                N, ftshape.begin(), 1,
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
    auto ftshape = ndarray::Vector<int, N+1>(shape);
    void *fp = detail::FFTWTraits<T>::forward(
                N, ftshape.begin()+1, ftshape[0],
                x.getData(), NULL, 1, x.template getStride<0>(),
                k.getData(), NULL, 1, k.template getStride<0>(),
                FFTW_MEASURE | FFTW_DESTROY_INPUT);
    if (fp == nullptr)
      throw std::exception();
    return Ptr(
        new FourierTransform(
            fp,
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
    auto ftshape = ndarray::Vector<int, N+1>(shape);
    return Ptr(
        new FourierTransform(
            detail::FFTWTraits<T>::inverse(
                N, ftshape.begin()+1, ftshape[0],
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
