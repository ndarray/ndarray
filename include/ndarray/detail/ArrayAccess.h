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
#ifndef NDARRAY_DETAIL_ArrayAccess_h_INCLUDED
#define NDARRAY_DETAIL_ArrayAccess_h_INCLUDED

/** 
 *  @file ndarray/detail/ArrayAccess.h
 *
 *  @brief Definitions for ArrayAccess
 */

#include "ndarray/ExpressionTraits.h"

namespace ndarray {
namespace detail {

template <typename Array_>
struct ArrayAccess {
    typedef typename ExpressionTraits< Array_ >::Element Element;
    typedef typename ExpressionTraits< Array_ >::Core Core;
    typedef typename ExpressionTraits< Array_ >::CorePtr CorePtr;

    static CorePtr const & getCore(Array_ const & array) {
        return array._core;
    }

    static Array_ construct(Element * data, CorePtr const & core) {
        return Array_(data, core);
    }

};

} // namespace detail
} // namespace ndarray

#endif // !NDARRAY_DETAIL_ArrayAccess_h_INCLUDED
