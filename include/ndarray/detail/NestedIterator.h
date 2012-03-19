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
#ifndef NDARRAY_DETAIL_NestedIterator_h_INCLUDED
#define NDARRAY_DETAIL_NestedIterator_h_INCLUDED

/** 
 *  @file ndarray/detail/NestedIterator.h
 *
 *  @brief Definition of NestedIterator.
 */

#include <boost/iterator/iterator_facade.hpp>
#include "ndarray_fwd.h"

namespace ndarray {
namespace detail {

/**
 *  @internal @brief Nested-array iterator class for Array with ND > 1.
 *
 *  While this iterator makes use of boost::iterator_facade, it
 *  reimplements the actual dereferencing operations (operator*,
 *  operator->) to return <b><tt>Reference const &</tt></b> and
 *  <b><tt>Reference const *</tt></b>, even though these are
 *  only convertible to the <b><tt>reference</tt></b> and
 *  <b><tt>pointer</tt></b> types associated with the iterator,
 *  not the types themselves.
 *
 *  @ingroup ndarrayInternalGroup
 */
template <typename T, int N, int C>
class NestedIterator : public boost::iterator_facade<
    NestedIterator<T,N,C>,
    typename ArrayTraits<T,N,C>::Value,
    boost::random_access_traversal_tag,
    typename ArrayTraits<T,N,C>::Reference
    >
{
public:
    typedef typename ArrayTraits<T,N,C>::Value Value;
    typedef typename ArrayTraits<T,N,C>::Reference Reference;
    
    Reference operator[](int n) const {
        Reference r(_ref);
        r._data += n * _stride;
        return r;
    }

    Reference const & operator*() const { return _ref; }

    Reference const * operator->() { return &_ref; }

    NestedIterator() : _ref(Value()), _stride(0) {}

    NestedIterator(Reference const & ref, int stride) : _ref(ref), _stride(stride) {}

    NestedIterator(NestedIterator const & other) : _ref(other._ref), _stride(other._stride) {}

    template <typename T_, int C_>
    NestedIterator(NestedIterator<T_,N,C_> const & other) : _ref(other._ref), _stride(other._stride) {}

    NestedIterator & operator=(NestedIterator const & other) {
        if (&other != this) {
            _ref._data = other._ref._data;
            _ref._core = other._ref._core;
            _stride = other._stride;
        }
        return *this;
    }

    template <typename T_, int C_>
    NestedIterator & operator=(NestedIterator<T_,N,C_> const & other) {
        _ref._data = other._ref._data;
        _ref._core = other._ref._core;
        _stride = other._stride;
        return *this;
    }

private:

    friend class boost::iterator_core_access;

    template <typename T_, int N_, int C_> friend class NestedIterator;

    Reference const & dereference() const { return _ref; }

    void increment() { _ref._data += _stride; }
    void decrement() { _ref._data -= _stride; }
    void advance(int n) { _ref._data += _stride * n; }

    template <typename T_, int C_>
    int distance_to(NestedIterator<T_,N,C_> const & other) const {
        return std::distance(_ref._data, other._ref._data) / _stride; 
    }

    template <typename T_, int C_>
    bool equal(NestedIterator<T_,N,C_> const & other) const {
        return _ref._data == other._ref._data;
    }

    Reference _ref;
    int _stride;
};

} // namespace detail
} // namespace ndarray

#endif // !NDARRAY_DETAIL_NestedIterator_h_INCLUDED
