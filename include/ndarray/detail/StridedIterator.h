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
#ifndef NDARRAY_DETAIL_StridedIterator_h_INCLUDED
#define NDARRAY_DETAIL_StridedIterator_h_INCLUDED

/** 
 *  @file ndarray/detail/StridedIterator.h
 *
 *  @brief Definition of StridedIterator.
 */

#include "ndarray_fwd.h"
#include <boost/iterator/iterator_facade.hpp>

namespace ndarray {
namespace detail {

/**
 *  @internal @brief Strided iterator for noncontiguous 1D arrays.
 *
 *  @ingroup ndarrayInternalGroup
 */
template <typename T>
class StridedIterator : public boost::iterator_facade<
    StridedIterator<T>, 
    T, boost::random_access_traversal_tag
    >
{
public:
    typedef T Value;
    typedef T & Reference;
    
    StridedIterator() : _data(0), _stride(0) {}

    StridedIterator(T * data, int stride) : _data(data), _stride(stride) {}

    StridedIterator(StridedIterator const & other) : _data(other._data), _stride(other._stride) {}

    template <typename U>
    StridedIterator(StridedIterator<U> const & other) : _data(other._data), _stride(other._stride) {
        BOOST_STATIC_ASSERT((boost::is_convertible<U*,T*>::value));
    }

    StridedIterator & operator=(StridedIterator const & other) {
        if (&other != this) {
            _data = other._data;
            _stride = other._stride;
        }
        return *this;
    }

    template <typename U>
    StridedIterator & operator=(StridedIterator<U> const & other) {
        BOOST_STATIC_ASSERT((boost::is_convertible<U*,T*>::value));
        _data = other._data;
        _stride = other._stride;
        return *this;
    }

private:

    friend class boost::iterator_core_access;

    template <typename OtherT> friend class StridedIterator;

    Reference dereference() const { return *_data; }

    void increment() { _data += _stride; }
    void decrement() { _data -= _stride; }
    void advance(int n) { _data += _stride * n; }

    template <typename U>
    int distance_to(StridedIterator<U> const & other) const {
        return std::distance(_data, other._data) / _stride; 
    }

    template <typename U>
    bool equal(StridedIterator<U> const & other) const {
        return _data == other._data;
    }

    T * _data;
    int _stride;

};

} // namespace detail
} // namespace ndarray

#endif // !NDARRAY_DETAIL_StridedIterator_h_INCLUDED
