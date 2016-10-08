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
#ifndef NDARRAY_detail_CompressedPair_hpp_INCLUDED
#define NDARRAY_detail_CompressedPair_hpp_INCLUDED

#include "ndarray/common.hpp"

namespace ndarray {
namespace detail {

template <typename First, typename Second>
class CompressedPair : private First, private Second {
public:
    typedef First first_type;
    typedef Second second_type;

    CompressedPair() : First(), Second() {}

    CompressedPair(first_type first, second_type second) :
        First(std::move(first)), Second(std::move(second))
    {}

    CompressedPair(CompressedPair const &) = default;

    CompressedPair(CompressedPair &&) = default;

    CompressedPair & operator=(CompressedPair const &) = default;

    CompressedPair & operator=(CompressedPair &&) = default;

    void swap(CompressedPair & other) {
        using std::swap;
        swap(first(), other.first());
        swap(second(), other.second());
    }

    first_type const & first() const { return *this; }

    first_type & first() { return *this; }

    second_type const & second() const { return *this; }

    second_type & second() { return *this; }

};

template <typename T1, typename T2>
inline void swap(CompressedPair<T1,T2> & a, CompressedPair<T1,T2> & b) {
    a.swap(b);
}

} // namespace detail
} // namespace ndarray

#endif // !NDARRAY_detail_CompressedPair_hpp_INCLUDED
