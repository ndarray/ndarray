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
#ifndef LSST_NDARRAY_TABLES_DETAIL_Iterator_h_INCLUDED
#define LSST_NDARRAY_TABLES_DETAIL_Iterator_h_INCLUDED

#include "lsst/ndarray_fwd.h"
#include "lsst/ndarray/tables/detail/TraitsAccess.h"

#include <boost/iterator/iterator_facade.hpp>

namespace lsst { namespace ndarray { namespace tables { namespace detail {

template <typename T>
class Iterator : public boost::iterator_facade< 
    Iterator<T>, 
    typename TraitsAccess<T>::Row_,
    boost::random_access_traversal_tag, 
    typename TraitsAccess<T>::Row_
    > 
{
public:

    typedef typename TraitsAccess<T>::Row_ Value;
    typedef Value Reference;

    Reference operator[](int n) const {
        return Reference(getRow()._n + n, getRow()._columns);
    }

    Reference const & operator*() const { return _ref; }

    Reference const * operator->() { return &_ref; }

    Iterator() : _ref() {}

    explicit Iterator(Reference const & ref) : _ref(ref) {}

    Iterator(Iterator const & other) : _ref(other._ref) {}

    template <typename U>
    Iterator(Iterator<U> const & other) : _ref(other._ref) {}
    
    Iterator & operator=(Iterator const & other) {
        if (&other != this) _ref = other._ref;
        return *this;
    }

    template <typename U>
    Iterator & operator=(Iterator<U> const & other) {
        if (&other != this) _ref = other._ref;
        return *this;
    }

private:

    friend class boost::iterator_core_access;

    template <typename U> friend class Iterator;

    Row<T> & getRow() { return _ref; }
    Row<T> const & getRow() const { return _ref; }

    Reference const & dereference() const { return _ref; }

    void increment() { ++getRow()._n; }
    void decrement() { --getRow()._n; }
    void advance(int n) { getRow()._n += n; }

    template <typename U>
    int distance_to(Iterator<U> const & other) const {
        return other.getRow()._n - getRow()._n; 
    }

    template <typename U>
    bool equal(Iterator<U> const & other) const {
        return other.getRow()._n == getRow()._n && other.getRow()._columns == other.getRow()._columns;
    }

    Reference _ref;
};


}}} // namespace lsst::ndarray::tables::detail

#endif // !LSST_NDARRAY_TABLES_DETAIL_Iterator_h_INCLUDED
