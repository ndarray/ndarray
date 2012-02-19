// -*- c++ -*-
/*
 * Copyright 2012, Jim Bosch and the LSST Corporation
 * 
 * ndarray is available under two licenses, both of which are described
 * more fully in other files that should be distributed along with
 * the code:
 * 
 *  - A simple BSD-style license (ndarray-bsd-license.txt); under this
 *    license ndarray is broadly compatible with essentially any other
 *    code.
 * 
 *  - As a part of the LSST data management software system, ndarray is
 *    licensed with under the GPL v3 (LsstLicenseStatement.txt).
 * 
 * These files can also be found in the source distribution at:
 * 
 * https://github.com/ndarray/ndarray
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
