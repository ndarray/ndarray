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
#ifndef LSST_NDARRAY_TABLES_DETAIL_FieldInfo_h_INCLUDED
#define LSST_NDARRAY_TABLES_DETAIL_FieldInfo_h_INCLUDED

#include "lsst/ndarray/tables_fwd.h"

namespace lsst { namespace ndarray { namespace tables { namespace detail {

template <typename Field>
struct FieldInfo {};

template <typename T, int N>
struct FieldInfo< Field<T,N> > {
    typedef typename boost::mpl::int_<N>::type ND;
    
    typedef Field<T,N> Field_;

    typedef typename boost::mpl::bool_<(N==0)>::type IsScalar;

    typedef lsst::ndarray::Array<T,N+1,N> ColumnValue;

    template <bool isConst>
    struct Qualified {
        typedef typename boost::mpl::if_c<isConst, T const, T>::type Element;
        typedef typename boost::mpl::if_<IsScalar, Element &, lsst::ndarray::ArrayRef<Element,N,N> >::type RowValue;
        typedef lsst::ndarray::ArrayRef<Element,N+1,N> TableValue;
    };

};

template <typename T, int N>
struct FieldInfo< Field<T,N> & > : public FieldInfo< Field<T,N> > {};

template <typename T, int N>
struct FieldInfo< Field<T,N> const & > : public FieldInfo< Field<T,N> > {};

}}}} // namespace lsst::ndarray::tables::detail

#endif // !LSST_NDARRAY_TABLES_DETAIL_FieldInfo_h_INCLUDED
