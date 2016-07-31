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
#ifndef NDARRAY_formatting_Vector_hpp_INCLUDED
#define NDARRAY_formatting_Vector_hpp_INCLUDED

#include <iostream>
#include "ndarray/Vector.hpp"
#include "ndarray/formatting/types.hpp"

namespace ndarray {

template <typename T, size_t N>
std::ostream & operator<<(std::ostream & os, Vector<T,N> const & v) {
    os << "Vector<" << type_string<T>() << ", " << N << ">{";
    if (N > 0) {
        os << v[0];
    }
    for (size_t i = 1; i < N; ++i) {
        os << ", " << v[i];
    }
    os << "}";
    return os;
}

} // ndarray

#endif // !NDARRAY_formatting_Vector_hpp_INCLUDED
