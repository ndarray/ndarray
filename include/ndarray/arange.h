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
#ifndef NDARRAY_arange_h_INCLUDED
#define NDARRAY_arange_h_INCLUDED

/** 
 *  @file ndarray/arange.h
 *
 *  @brief Expression classes to generate regularly-spaced ranges of values.
 */

#include "ndarray/vectorize.h"

#include <boost/iterator/counting_iterator.hpp>

namespace ndarray {

/**
 *  @internal @brief ExpressionTraits specialization for CountingExpression.
 *
 *  @ingroup ndarrayInternalGroup
 */
template <>
struct ExpressionTraits<detail::CountingExpression> {
    typedef std::size_t Element;
    typedef boost::mpl::int_<1> ND;
    typedef boost::counting_iterator<std::size_t> Iterator;
    typedef std::size_t Value;
    typedef std::size_t Reference;
};

namespace detail {

/**
 *  @internal @class CountingExpression
 *  @brief Expression that simply iterates over integer values.
 *
 *  @ingroup ndarrayInternalGroup
 */
class CountingExpression : public ExpressionBase<CountingExpression> {
public:
    typedef ExpressionTraits<CountingExpression>::Element Element;
    typedef ExpressionTraits<CountingExpression>::ND ND;
    typedef ExpressionTraits<CountingExpression>::Iterator Iterator;
    typedef ExpressionTraits<CountingExpression>::Value Value;
    typedef ExpressionTraits<CountingExpression>::Reference Reference;
    typedef Vector<std::size_t,1> Index;
    
    CountingExpression(std::size_t stop=0) : _stop(stop) { NDARRAY_ASSERT(stop >= 0); }

    Reference operator[](std::size_t n) const {
        return n;
    }

    Iterator begin() const {
        return Iterator(0);
    }

    Iterator end() const {
        return Iterator(_stop);
    }

    template <std::size_t P> std::size_t getSize() const {
        BOOST_STATIC_ASSERT(P==0);
        return _stop;
    }

    Index getShape() const {
        return makeVector(_stop);
    }

private:
    std::size_t _stop;
};

template <typename T>
class RangeTransformer {
    T _offset;
    T _scale;
public:
    typedef std::size_t argument_type;
    typedef T result_type;

    explicit RangeTransformer(T const & offset, T const & scale) : _offset(offset), _scale(scale) {}

    T operator()(std::size_t n) const { return static_cast<T>(n) * _scale + _offset; }
};

} // namespace detail

/// @brief Create 1D Expression that contains integer values in the range [0,stop).
inline detail::CountingExpression arange(std::size_t stop) {
    return detail::CountingExpression(stop);
}

/// @brief Create 1D Expression that contains integer values in the range [start,stop) with increment step.
template <typename T>
detail::UnaryOpExpression< detail::CountingExpression, detail::RangeTransformer<T> >
arange(T start, T stop, T step = 1) {
    NDARRAY_ASSERT(step != 0);
    T const diff = stop - start;
    NDARRAY_ASSERT((diff > 0 && step > 0) || (diff < 0 && step < 0));
    std::size_t const size = diff/step;
    return vectorize(
        detail::RangeTransformer<T>(start,step),
        detail::CountingExpression(size)
    );
}

} // namespace ndarray

#endif // !NDARRAY_arange_h_INCLUDED
