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
#include <stdexcept>
#include <sstream>

namespace ndarray {

typedef std::uint8_t byte_t;
typedef std::size_t size_t;
typedef std::ptrdiff_t offset_t;

enum class MemoryOrder {
    ROW_MAJOR,
    COL_MAJOR
};

namespace detail {

template <size_t N> class Layout;

template <typename T, size_t N, offset_t C> struct ArrayTraits;

template <typename T, size_t N, offset_t C> struct IterTraits;

template <typename T, size_t N> class ArrayImpl;

template <typename T, size_t N, offset_t C> class IterImpl;

} // namespace detail

template <typename T, size_t N, offset_t C> class ArrayBase;

template <typename T, size_t N, offset_t C=1> class Array;

template <typename T, size_t N, offset_t C=1> class ArrayRef;

template <typename T, size_t N, offset_t C> class Iter;

template <typename T, size_t N> class Vector;

template <typename T> class DType;

class Manager;

class NoncontiguousError : public std::logic_error {

    std::string format(offset_t actual, offset_t required) {
        std::ostringstream s;
        if (required > 0) {
            s << "Template requires at least " << required
              << " row-major contiguous dimensions;"
              << " strides only have " << actual;
        } else {
            s << "Template requires at least " << (-required)
              << " column-major contiguous dimensions;"
              << " strides only have " << (-actual);
        }
        return s.str();
    }

public:

    explicit NoncontiguousError(char const * msg) : std::logic_error(msg) {}

    explicit NoncontiguousError(offset_t actual, offset_t required) :
        std::logic_error(format(actual, required))
    {}

};

} // ndarray

#endif // !NDARRAY_common_hpp_INCLUDED
