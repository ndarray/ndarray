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
#include <sstream>
#include "ndarray/exceptions.hpp"

namespace ndarray {

std::string NoncontiguousError::format(offset_t actual, offset_t required) {
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

} // ndarray
