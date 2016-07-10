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
#ifndef NDARRAY_dtype_hpp_INCLUDED
#define NDARRAY_dtype_hpp_INCLUDED

#include <memory>
#include <type_traits>

#include "ndarray/common.hpp"

namespace ndarray {

namespace detail {

template <typename T>
struct get_dtype {
    typedef DType<T> type;
};

template <typename T>
struct get_dtype<T const> {
    typedef DType<T> type;
};

template <typename T>
using get_dtype_t = typename get_dtype<T>::type;

} // detail


template <typename T>
class DType {
public:

    static_assert(!std::is_reference<T>::value, "reference dtypes not supported");
    static_assert(!std::is_const<T>::value, "const dtypes not supported");

    typedef T value_type;
    typedef T & reference;
    typedef T const & const_reference;
    typedef T * pointer;
    typedef T const & const_pointer;
    typedef Iter<T,1> iterator;
    typedef Iter<T const,1> const_iterator;
    typedef std::is_pod<T> is_pod;

    DType() {}

    DType(DType const & other) = default;
    DType(DType && other) = default;

    template <typename U>
    DType(DType<U> const & other) {
        static_assert(
            std::is_convertible<U*,T*>::value || (is_pod::value && DType<U>::is_pod::value),
            "Cannot reinterpet types unless both are POD"
        );
    }

    DType & operator=(DType const & other) = default;
    DType & operator=(DType && other) = default;

    void swap(DType & other) {}

    size_t nbytes() const { return sizeof(T); }

    bool operator==(DType const & other) const { return true; }
    bool operator!=(DType const & other) const { return false; }

};

} // ndarray

#endif // !NDARRAY_dtype_hpp_INCLUDED
