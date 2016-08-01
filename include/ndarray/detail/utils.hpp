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
#ifndef NDARRAY_detail_utils_hpp_INCLUDED
#define NDARRAY_detail_utils_hpp_INCLUDED

#include "ndarray/common.hpp"

namespace ndarray {
namespace detail {

template <typename Iterator1, typename Iterator2, typename Function>
void for_each_zip(
    Iterator first1,
    Iterator last1,
    Iterator first2,
    Function binary_op
) {
    while (first1 != last1) {
        binary_op(first1, first2);
        ++first1;
        ++first2;
    }
}

} // namespace detail
} // namespace ndarray

#endif // !NDARRAY_detail_utils_hpp_INCLUDED
