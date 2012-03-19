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
#ifndef NDARRAY_tables_fwd_h_INCLUDED
#define NDARRAY_tables_fwd_h_INCLUDED

/**
 * @file ndarray/tables_fwd.h 
 *
 * @brief Forward declarations for ndarray Tables library.
 *
 *  \note This file is not included by the main "ndarray.h" header file.
 */

#include "ndarray_fwd.h"
#include <boost/type_traits/is_const.hpp>
#include <boost/type_traits/remove_const.hpp>

namespace ndarray { namespace tables {

template <int N> struct Index {};

template <typename T, int N=0> struct Field;
template <typename T> class Layout;

template <typename T> class Row;
template <typename T> class Table;

namespace detail {

template <typename T, bool isConst = boost::is_const<T>::value> struct TraitsAccess;
template <typename Field_> struct FieldInfo;
template <typename T> class Columns;
template <typename T> class Iterator;

} // namespace detail

template <typename T>
struct Traits {
    typedef typename T::FieldSequence FieldSequence;
    typedef Row<T> Record;
    typedef Row<T> ConstRecord;
};

}} // namespace ndarray::tables

#endif // !NDARRAY_tables_fwd_h_INCLUDED
