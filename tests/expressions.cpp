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

#include "ndarray/expressions/Strided.hpp"

using namespace ndarray;
using namespace ndarray::expressions;

TEST_CASE("expressions", "[expressions]") {
    std::array<int, 2*3*4> values = {0};
    std::iota(values.begin(), values.end(), 1);
    std::array<Size, 3> shape = {2, 3, 4};
    std::array<Offset, 3> strides = {12, 4, 1};
    int n = 1;
    StridedExpression<int, 3> expr(values.data(), shape, strides);
    auto t0 = expr.traverse();
    for (Size k0 = 0; k0 < shape[0]; ++k0) {
        auto t1 = t0.evaluate();
        for (Size k1 = 0; k1 < shape[1]; ++k1) {
            auto t2 = t1.evaluate();
            for (Size k2 = 0; k2 <shape[2]; ++k2) {
                REQUIRE(t2.evaluate() == n);
                ++n;
                t2.increment();
            }
            t1.increment();
        }
        t0.increment();
    }
}
