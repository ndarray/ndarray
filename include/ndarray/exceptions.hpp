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
#ifndef NDARRAY_exceptions_hpp_INCLUDED
#define NDARRAY_exceptions_hpp_INCLUDED

#include <stdexcept>
#include "fmt/format.h"
#include "ndarray/common.hpp"

namespace ndarray {

class NoncontiguousError : public std::logic_error {

    static std::string format(Size actual, Offset required) {
        static constexpr auto s = "Template parameters require at least {:d} {:s} "
                                  "contiguous parameters; array only has {:d}";
        if (required > 0) {
            return fmt::format(s, required, "row-major", actual);
        }
        return fmt::format(s, -required, "column-major", actual);
    }

public:

    explicit NoncontiguousError(char const * msg) : std::logic_error(msg) {}

    explicit NoncontiguousError(Size actual, Offset required) :
        std::logic_error(format(actual, required))
    {}

};

} // ndarray

#endif // !NDARRAY_exceptions_hpp_INCLUDED
