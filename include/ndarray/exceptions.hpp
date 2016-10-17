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
#ifndef NDARRAY_exceptions_hpp_INCLUDED
#define NDARRAY_exceptions_hpp_INCLUDED

#include <stdexcept>
#include <sstream>
#include "ndarray/common.hpp"

namespace ndarray {

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

#endif // !NDARRAY_exceptions_hpp_INCLUDED
