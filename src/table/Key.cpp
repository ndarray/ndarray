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

#include "ndarray/table/Key.hpp"

namespace ndarray {

std::string TypeError::format(
    std::string const & desired,
    std::string const & actual
) {
    std::ostringstream s;
    s << "Key has type '" << actual << "'', not '" << desired << "'.";
    return s.str();
}

} // ndarray
