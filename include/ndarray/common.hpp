// -*- c++ -*-
/*
 * Copyright (c) 2010-2018, Jim Bosch
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

typedef std::uint8_t Byte;
typedef std::size_t Size;
typedef std::ptrdiff_t Offset;

enum class MemoryOrder {
    ROW_MAJOR,
    COL_MAJOR
};

namespace detail {

template <Size N> class Layout;

template <Size N> struct ArrayImpl;

template <MemoryOrder order> class OrderTag {};
using RowMajorTag = OrderTag<MemoryOrder::ROW_MAJOR>;
using ColMajorTag = OrderTag<MemoryOrder::COL_MAJOR>;

template <typename T> class TypeTag {};

} // namespace detail

} // ndarray

#endif // !NDARRAY_common_hpp_INCLUDED
