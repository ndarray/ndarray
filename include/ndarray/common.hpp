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
#ifndef NDARRAY_common_hpp_INCLUDED
#define NDARRAY_common_hpp_INCLUDED

#include <cstdint>

namespace ndarray {

typedef std::uint8_t byte_t;
typedef std::uint32_t dtype_size;
typedef std::int32_t dtype_offset;
typedef std::size_t size_t;
typedef std::ptrdiff_t offset_t;

enum class MemoryOrder {
    ROW_MAJOR,
    COL_MAJOR;
};

namespace detail {

template <int N> class layout;

template <typename T, int N> struct array_traits;

} // namespace detail

template <typename T, int N> class array;

template <typename T, int N> class array_val;

template <typename T, int N> class array_ref;

template <typename T> class dtype;

class manager;

template <typename T> class record;

template <typename T> class record_ref;

class schema;

} // ndarray

#endif // !NDARRAY_common_hpp_INCLUDED
