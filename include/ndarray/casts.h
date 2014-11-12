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
#ifndef NDARRAY_casts_h_INCLUDED
#define NDARRAY_casts_h_INCLUDED

/** 
 *  @file ndarray/casts.h
 *
 *  @brief Specialized casts for Array.
 */
#include <cstddef>
 
#include "ndarray/Array.h"
#include "ndarray/ArrayRef.h"
#include <boost/type_traits/add_const.hpp>
#include <boost/type_traits/remove_const.hpp>
#include <boost/mpl/comparison.hpp>
#include <boost/static_assert.hpp>

namespace ndarray {
namespace detail {

template <typename Array_>
struct ComplexExtractor {
    typedef typename ExpressionTraits<Array_>::Element ComplexElement;
    typedef typename boost::remove_const<ComplexElement>::type ComplexValue;
    typedef typename ExpressionTraits<Array_>::ND ND;
    BOOST_STATIC_ASSERT( boost::is_complex<ComplexValue>::value );
    typedef typename ComplexValue::value_type RealValue;
    typedef typename boost::mpl::if_<
        boost::is_const<ComplexElement>, RealValue const, RealValue
        >::type RealElement;
    typedef ArrayRef<RealElement,ND::value,0> Result;
    typedef detail::ArrayAccess<Result> Access;
    typedef Vector<std::size_t,ND::value> Index;

    static inline Result apply(Array_ const & array, std::size_t offset) {
        return Access::construct(
            reinterpret_cast<RealElement*>(array.getData()) + offset,
            Access::Core::create(array.getShape(), array.getStrides() * 2, array.getManager())
        );
    }
};

} // namespace detail

/// @addtogroup MainGroup
/// @{

/**
 *  Convert an Array with a const data type to an array
 *  with a non-const data type.
 */
template <typename T_, typename T, int N, int C>
Array<T_,N,C>
const_array_cast(Array<T,N,C> const & array) {
    return detail::ArrayAccess< Array<T_,N,C> >::construct(
        const_cast<T_*>(array.getData()),
        detail::ArrayAccess< Array<T,N,C> >::getCore(array)
    );
}

/**
 *  Convert an Array to a type with more guaranteed
 *  row-major-contiguous dimensions with no checking.
 */
template <int C_, typename T, int N, int C>
Array<T,N,C_>
static_dimension_cast(Array<T,N,C> const & array) {
    return detail::ArrayAccess< Array<T,N,C_> >::construct(
        array.getData(),
        detail::ArrayAccess< Array<T,N,C> >::getCore(array)
    );
}

/**
 *  Convert an Array to a type with more guaranteed
 *  row-major-contiguous dimensions, if the strides
 *  of the array match the desired number of RMC
 *  dimensions.  If the cast fails, an empty Array
 *  is returned.
 */
template <int C_, typename T, int N, int C>
Array<T,N,C_>
dynamic_dimension_cast(Array<T,N,C> const & array) {
    Vector<std::size_t,N> shape = array.getShape();
    Vector<std::size_t,N> strides = array.getStrides();
    if (C_ >= 0) {
        std::size_t n = 1;
        for (int i=1; i <= C_; ++i) {
            if (strides[static_cast<size_t>(N-i)] != n) return Array<T,N,C_>();
            n *= shape[N-i];
        }
    } else {
        std::size_t n = 1;
        for (std::size_t i=0; i < static_cast<size_t>(-C_); ++i) {
            if (strides[i] != n) return Array<T,N,C_>();
            n *= strides[i];
        }
    }
    return static_dimension_cast<C_>(array);
}

/**
 *  @brief Return an ArrayRef view into the real part of a complex array.
 */
template <typename Array_>
typename detail::ComplexExtractor<Array_>::Result
getReal(Array_ const & array) {
    return detail::ComplexExtractor<Array_>::apply(array, 0);
}

/**
 *  @brief Return an ArrayRef view into the imaginary part of a complex array.
 */
template <typename Array_>
typename detail::ComplexExtractor<Array_>::Result
getImag(Array_ const & array) {
    return detail::ComplexExtractor<Array_>::apply(array, 1);
}

/**
 *  @brief Create a view into an array with trailing contiguous dimensions merged.
 *
 *  The first template parameter sets the dimension of the output array and must
 *  be specified directly.  Only row-major contiguous dimensions can be flattened.
 */
template <int Nf, typename T, int N, int C>
inline typename boost::enable_if_c< ((C+Nf-N)>=1), ArrayRef<T,Nf,(C+Nf-N)> >::type
flatten(Array<T,N,C> const & input) {
    typedef detail::ArrayAccess< ArrayRef<T,Nf,(C+Nf-N)> > Access;
    typedef typename Access::Core Core;
    BOOST_STATIC_ASSERT(C+Nf-N >= 1);
    Vector<std::size_t,N> oldShape = input.getShape();
    Vector<std::size_t,Nf> newShape = oldShape.template first<Nf>();
    for (std::size_t n=static_cast<size_t>(Nf); n<static_cast<size_t>(N); ++n)
        newShape[Nf-1] *= oldShape[n];
    Vector<std::size_t,Nf> newStrides = input.getStrides().template first<Nf>();
    newStrides[static_cast<size_t>(Nf-1)] = 1;
    return Access::construct(input.getData(), Core::create(newShape, newStrides, input.getManager()));
}

/// @}

} // namespace ndarray

#endif // !NDARRAY_casts_h_INCLUDED
