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
#ifndef NDARRAY_detail_compressed_pair_hpp_INCLUDED
#define NDARRAY_detail_compressed_pair_hpp_INCLUDED

#include "ndarray/common.hpp"
#include "ndarray/dtype.hpp"

namespace ndarray {
namespace detail {

template <typename First, typename Second>
class compressed_pair : private First, private Second {
public:
    typedef First first_type;
    typedef Second second_type;

    compressed_pair(first_type first, second_dtype second) :
        First(std::move(first)), Second(std::move(second))
    {}

    compressed_pair(compressed_pair const &) = default;

    compressed_pair(compressed_pair &&) = default;

    compressed_pair & operator=(compressed_pair const &) = default;

    compressed_pair & operator=(compressed_pair &&) = default;

    first_type const & first() const { return *this; }

    first_type & first() { return *this; }

    second_type const & second() const { return *this; }

    second_type & second() { return *this; }

};

} // namespace detail
} // namespace ndarray

#endif // !NDARRAY_detail_compressed_pair_hpp_INCLUDED
