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
#ifndef LSST_NDARRAY_TABLES_DETAIL_TraitsAccess_h_INCLUDED
#define LSST_NDARRAY_TABLES_DETAIL_TraitsAccess_h_INCLUDED

#include "lsst/ndarray/tables_fwd.h"
#include "lsst/ndarray/tables/detail/FieldInfo.h"

#include <boost/intrusive_ptr.hpp>

namespace lsst { namespace ndarray { namespace tables { namespace detail {

template <typename T>
struct TraitsAccess<T,false> {
    typedef typename Traits<T>::FieldSequence FieldSequence;
    typedef Layout<T> Layout_;
    typedef Row<T> Row_;
    typedef Table<T> Table_;
    typedef unsigned char Raw;
    typedef detail::Columns<T> Columns_;
    typedef boost::intrusive_ptr< detail::Columns<T> > ColumnsPtr;

    template <int N>
    struct Fields {
        typedef typename boost::fusion::result_of::value_at_c<FieldSequence,N>::type Type;
        typedef FieldInfo<Type> Info;
        typedef typename Info::template Qualified<false> Qualified;
    };

};

template <typename T>
struct TraitsAccess<T,true> {
    typedef typename boost::remove_const<T>::type U;
    typedef typename Traits<T>::FieldSequence FieldSequence;
    typedef Layout<U> Layout_;
    typedef Row<T> Row_;
    typedef Table<T> Table_;
    typedef unsigned char const Raw;
    typedef detail::Columns<U> Columns_;
    typedef boost::intrusive_ptr< detail::Columns<U> > ColumnsPtr;

    template <int N>
    struct Fields {
        typedef typename boost::fusion::result_of::value_at_c<FieldSequence,N>::type Type;
        typedef FieldInfo<Type> Info;
        typedef typename Info::template Qualified<true> Qualified;
    };
};

}}} // namespace lsst::ndarray::tables::detail

#endif // !LSST_NDARRAY_TABLES_DETAIL_TraitsAccess_h_INCLUDED
