// -*- c++ -*-
/*
 * Copyright (c) 2010-2016, Jim Bosch
 * All rights reserved.
 *
 * ndarray is distributed under a simple BSD-like license;
 * see the LICENSE file that should be present in the root
 * of the source distribution, or alternately available at:
 * https://github.com/ndarray/ndarray
 */
#ifndef NDARRAY_any_hpp_INCLUDED
#define NDARRAY_any_hpp_INCLUDED

#include "ndarray/common.hpp"

#if __has_include(<experimental/any>)

#   include <experimental/any>
    namespace ndarray {
        using any = std::experimental::any;
        using any_cast = std::experimental::any_cast;
    }

#elif defined(NDARRAY_USE_BOOST_ANY)

#   include "boost/any.hpp"
    namespace ndarray {
        using any = boost::any;
        using any_cast = boost::any_cast;
    }

#else

#error "This feature requires either std::experimental::any or boost::any."

#endif

#endif // !NDARRAY_any_hpp_INCLUDED
