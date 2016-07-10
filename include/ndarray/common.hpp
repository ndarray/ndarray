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
typedef std::size_t size_t;
typedef std::ptrdiff_t offset_t;

enum class MemoryOrder {
    ROW_MAJOR,
    COL_MAJOR
};

namespace detail {

template <int N> class Layout;

template <typename T, int N> struct ArrayTraits;

template <typename T, int N> struct IterTraits;

template <typename T, int N> class ArrayImpl;

template <typename T, int N> class IterImpl;

} // namespace detail

template <typename T, int N> class ArrayBase;

template <typename T, int N> class Array;

template <typename T, int N> class ArrayRef;

template <typename T, int N> class Iter;

template <typename T> class DType;

class Manager;

template <typename T> class record;

template <typename T> class record_ref;

class schema;

} // ndarray

#endif // !NDARRAY_common_hpp_INCLUDED
