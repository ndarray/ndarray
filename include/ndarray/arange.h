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
    typedef int Element;
    typedef boost::mpl::int_<1> ND;
    typedef boost::counting_iterator<int> Iterator;
    typedef int Value;
    typedef int Reference;
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
    typedef Vector<int,1> Index;
    
    CountingExpression(int stop=0) : _stop(stop) { NDARRAY_ASSERT(stop >= 0); }

    Reference operator[](int n) const {
        return n;
    }

    Iterator begin() const {
        return Iterator(0);
    }

    Iterator end() const {
        return Iterator(_stop);
    }

    template <int P> int getSize() const {
        BOOST_STATIC_ASSERT(P==0);
        return _stop;
    }

    Index getShape() const {
        return makeVector(_stop);
    }

private:
    int _stop;
};

template <typename T>
class RangeTransformer {
    T _offset;
    T _scale;
public:
    typedef int argument_type;
    typedef T result_type;

    explicit RangeTransformer(T const & offset, T const & scale) : _offset(offset), _scale(scale) {}

    T operator()(int n) const { return static_cast<T>(n) * _scale + _offset; }
};

} // namespace detail

/// @brief Create 1D Expression that contains integer values in the range [0,stop).
inline detail::CountingExpression arange(int stop) {
    return detail::CountingExpression(stop);
}

/// @brief Create 1D Expression that contains integer values in the range [start,stop) with increment step.
inline detail::UnaryOpExpression< detail::CountingExpression, detail::RangeTransformer<int> >
arange(int start, int stop, int step = 1) {
    NDARRAY_ASSERT(step != 0);
    int size = stop - start;
    if (step < -1) ++size;
    if (step > 1) --size;
    size /= step;
    return vectorize(
        detail::RangeTransformer<int>(start,step),
        detail::CountingExpression(size)
    );
}

} // namespace ndarray

#endif // !NDARRAY_arange_h_INCLUDED
