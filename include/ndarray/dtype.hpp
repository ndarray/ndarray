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

#include "ndarray/commmon.hpp"

namespace ndarray {

namespace detail {

template <typename T>
struct get_dtype {
    typedef dtype<T> type;
};

template <typename T>
struct get_dtype<T const> {
    typedef dtype<T> type;
};

template <typename T>
using get_dtype_t = typename get_dtype<T>::type;

} // detail


template <typename T>
class dtype {
public:

    static_assert(!std::is_reference<T>::value, "reference dtypes not supported");
    static_assert(!std::is_const<T>::value, "const dtypes not supported");

    typedef T value;
    typedef T & reference;
    typedef T const & const_reference;
    typedef T * pointer;
    typedef T const & const_pointer;
    typedef iterator<T,0> iterator;
    typedef iterator<T const,0> const_iterator;
    typedef std::is_pod<Value> is_pod;
    typedef std::false_type is_proxy;

    dtype() {}

    dtype(dtype const & other) = default;
    dtype(dtype && other) = default;

    template <typename U>
    dtype(dtype<U> const & other) {
        static_assert(
            is_convertible<U*,T*>::value || (is_pod::value && dtype<U>::is_pod::value),
            "Cannot reinterpet types unless both are POD"
        );
    }

    dtype & operator=(dtype const & other) = default;
    dtype & operator=(dtype && other) = default;

    void swap(dtype & other) {}

    dtype_size nbytes() const { return sizeof(T); }

    reference make_reference(
        byte_t * data,
        std::shared_ptr<manager> const &
    ) const {
        return *reinterpret_cast<T*>(data);
    }

    const_reference make_const_reference(
        byte_t const * data,
        std::shared_ptr<manager> const &
    ) const {
        return *reinterpret_cast<T const*>(data);
    }

    pointer make_pointer(
        byte_t * data,
        std::shared_ptr<manager> const &
    ) const {
        return reinterpret_cast<T*>(data);
    }

    const_pointer make_const_pointer(
        byte_t const * data,
        std::shared_ptr<manager> const &
    ) const {
        return reinterpret_cast<T const*>(data);
    }

};

} // ndarray

#endif // !NDARRAY_dtype_hpp_INCLUDED
