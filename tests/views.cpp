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
#include <numeric>
#include "catch2/catch.hpp"

#define NDARRAY_ASSERT_AUDIT_ENABLED true

#include "ndarray/views.hpp"

using namespace ndarray::views;


TEST_CASE("views", "[views]") {
    auto v = view(all)(1, 2)(begin, 5)(3, end)(1, 2, -unit)(newaxis)(3, 4, 2);
}
