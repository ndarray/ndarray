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
// TODO: replace type_string magic with explicit specializations
#include "ndarray/formatting/types.hpp"

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
    typedef T const * const_pointer;
    static constexpr bool is_pod = std::is_pod<T>::value;
    static constexpr bool is_direct = true;

    DType() {}

    DType(DType const & other) = default;
    DType(DType && other) = default;

    template <typename U>
    DType(DType<U> const & other) {
        static_assert(
            std::is_convertible<U*,T*>::value || (is_pod && DType<U>::is_pod),
            "Cannot reinterpret types unless both are POD"
        );
    }

    DType & operator=(DType const & other) = default;
    DType & operator=(DType && other) = default;

    void swap(DType & other) {}

    size_t alignment() const { return alignof(T); }

    size_t nbytes() const { return sizeof(T); }

    static std::string const & name() {
        // TODO: replace type_string magic with explicit specializations
        static std::string x = type_string<T>();
        return x;
    }

    bool operator==(DType const & other) const { return true; }
    bool operator!=(DType const & other) const { return false; }

    void initialize(byte_t * buffer) const {
        new (buffer) T;
    }

    void destroy(byte_t * buffer) const {
        reinterpret_cast<T*>(buffer)->~T();
    }

    reference make_reference_at(
        byte_t * buffer,
        std::shared_ptr<Manager> const &
    ) const {
        return *reinterpret_cast<T*>(buffer);
    }

    const_reference make_const_reference_at(
        byte_t const * buffer,
        std::shared_ptr<Manager> const &
    ) const {
        return *reinterpret_cast<T*>(buffer);
    }

};


template <typename T>
void swap(DType<T> & a, DType<T> & b) {
    a.swap(b);
}

} // ndarray

#endif // !NDARRAY_dtype_hpp_INCLUDED
