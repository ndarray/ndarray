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
#ifndef LSST_NDARRAY_TABLES_Field_h_INCLUDED
#define LSST_NDARRAY_TABLES_Field_h_INCLUDED

#include "lsst/ndarray/tables_fwd.h"
#include "lsst/ndarray/Vector.h"

namespace lsst { namespace ndarray { namespace tables {

template <typename T, int N>
struct Field {
    typedef T Element;
    typedef typename boost::mpl::int_<N>::type ND;
    typedef typename boost::mpl::bool_<(N==0)>::type IsScalar;

    std::string name;
    Vector<int,N> shape;
    int offset;

    template <typename T1, int N1>
    bool operator==(Field<T1,N1> const & other) const { return false; }

    template <typename T1, int N1>
    bool operator!=(Field<T1,N1> const & other) const { return true; }

    bool operator==(Field<T,N> const & other) const {
        return name == other.name && shape == other.shape 
            && (offset == other.offset || offset < 0 || other.offset < 0);
    }

    bool operator!=(Field<T,N> const & other) const {
        return !this->operator==(other);
    }

    Field() : name(), shape(), offset(-1) {}

    Field(Field const & other) : name(other.name), shape(other.shape), offset(other.offset) {}

    Field & operator=(Field const & other) {
        if (&other != this) {
            name = other.name;
            shape = other.shape;
            offset = other.offset;
        }
        return *this;
    }
};

}}} // namespace lsst::ndarray::tables

#endif // !LSST_NDARRAY_TABLES_Field_h_INCLUDED
