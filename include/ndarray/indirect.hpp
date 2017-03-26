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

#include "ndarray/DType.hpp"

namespace ndarray {


template <typename T>
class IndirectDTypeBase {
public:

    static_assert(!std::is_reference<T>::value, "reference dtypes not supported");
    static_assert(!std::is_const<T>::value, "const dtypes not supported");

    typedef T value_type;
    typedef T reference;
    typedef T const_reference;
    typedef std::unique_ptr<T> pointer;
    typedef std::unique_ptr<T> const_pointer;
    static constexpr bool is_pod = false;
    static constexpr bool is_direct = false;

    IndirectDTypeBase() {}

    IndirectDTypeBase(IndirectDTypeBase const & other) = default;
    IndirectDTypeBase(IndirectDTypeBase && other) = default;

    IndirectDTypeBase & operator=(IndirectDTypeBase const & other) = default;
    IndirectDTypeBase & operator=(IndirectDTypeBase && other) = default;
};


} // ndarray

#endif // !NDARRAY_dtype_hpp_INCLUDED
