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
#ifndef NDARRAY_formatting_types_hpp_INCLUDED
#define NDARRAY_formatting_types_hpp_INCLUDED

#include <string>
#include <typeinfo>
#ifndef _MSC_VER
#include <cxxabi.h>
#endif

namespace ndarray {

template <typename T>
std::string type_string() {
#if defined(_MSC_VER_)
    return typeid(T).name();
#else
    std::string sig = __PRETTY_FUNCTION__;
    size_t start = sig.find("T = ") + 4;
# if defined(__clang__)
    size_t end = sig.find(']', start);
# elif defined(__GNUC__)
    size_t end = sig.find(';', start);
# endif
#endif
    return sig.substr(start, end - start);
}

} // ndarray

#endif // !NDARRAY_formatting_types_hpp_INCLUDED
