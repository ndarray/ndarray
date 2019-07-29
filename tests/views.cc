// -*- c++ -*-
/*
 * Copyright (c) 2010-2012, Jim Bosch
 * All rights reserved.
 *
 * ndarray is distributed under a simple BSD-like license;
 * see the LICENSE file that should be present in the root
 * of the source distribution, or alternately available at:
 * https://github.com/ndarray/ndarray
 */
#include "ndarray.h"

#define BOOST_TEST_MODULE views
#include "boost/test/included/unit_test.hpp"

template <typename T, int N, int C>
int templateC(ndarray::ArrayRef<T,N,C> const &) { return C; }

template <typename T, int N, int C>
int strideC(ndarray::ArrayRef<T,N,C> const & v) {
    if (C >= 0) {
        int c = 0;
        int stride = 1;
        for (int n = N-1; n >= 0; --n, ++c) {
            if (v.getStrides()[n] != stride) break;
            stride *= v.getShape()[n];
        }
        return c;
    } else {
        int c = 0;
        int stride = 1;
        for (int n = 0; n < N; ++n, --c) {
            if (v.getStrides()[n] != stride) break;
            stride *= v.getShape()[n];
        }
        return c;
    }
}

#define CHECK_VIEW_RMC(V)                       \
    BOOST_CHECK_EQUAL(templateC(V), strideC(V))

BOOST_AUTO_TEST_CASE(view2p2) {

    ndarray::Array<float,2,2> array = ndarray::allocate(5,4);

    CHECK_VIEW_RMC(array[ndarray::view(1,3,2)(0,2,2)]);   // (slice, slice)
    CHECK_VIEW_RMC(array[ndarray::view(1,3,2)(0,2)]);     // (slice, range)
    CHECK_VIEW_RMC(array[ndarray::view(1,3,2)()]);        // (slice, full)
    CHECK_VIEW_RMC(array[ndarray::view(1,3,2)(1)]);       // (slice, scalar)

    CHECK_VIEW_RMC(array[ndarray::view(1,3)(0,2,2)]);     // (range, slice)
    CHECK_VIEW_RMC(array[ndarray::view(1,3)(0,2)]);       // (range, range)
    CHECK_VIEW_RMC(array[ndarray::view(1,3)()]);          // (range, full)
    CHECK_VIEW_RMC(array[ndarray::view(1,3)(1)]);         // (range, scalar)

    CHECK_VIEW_RMC(array[ndarray::view()(0,2,2)]);        // (range, slice)
    CHECK_VIEW_RMC(array[ndarray::view()(0,2)]);          // (range, range)
    CHECK_VIEW_RMC(array[ndarray::view()()]);             // (range, full)
    CHECK_VIEW_RMC(array[ndarray::view()(1)]);            // (range, scalar)

    CHECK_VIEW_RMC(array[ndarray::view(2)(0,2,2)]);       // (scalar, slice)
    CHECK_VIEW_RMC(array[ndarray::view(2)(0,2)]);         // (scalar, range)
    CHECK_VIEW_RMC(array[ndarray::view(2)()]);            // (scalar, full)

}

BOOST_AUTO_TEST_CASE(view2p1) {

    ndarray::Array<float,2,2> parent = ndarray::allocate(10,4);
    ndarray::Array<float,2,1> array = parent[ndarray::view(0,10,2)()];

    CHECK_VIEW_RMC(array[ndarray::view(1,3,2)(0,2,2)]);   // (slice, slice)
    CHECK_VIEW_RMC(array[ndarray::view(1,3,2)(0,2)]);     // (slice, range)
    CHECK_VIEW_RMC(array[ndarray::view(1,3,2)()]);        // (slice, full)
    CHECK_VIEW_RMC(array[ndarray::view(1,3,2)(1)]);       // (slice, scalar)

    CHECK_VIEW_RMC(array[ndarray::view(1,3)(0,2,2)]);     // (range, slice)
    CHECK_VIEW_RMC(array[ndarray::view(1,3)(0,2)]);       // (range, range)
    CHECK_VIEW_RMC(array[ndarray::view(1,3)()]);          // (range, full)
    CHECK_VIEW_RMC(array[ndarray::view(1,3)(1)]);         // (range, scalar)

    CHECK_VIEW_RMC(array[ndarray::view()(0,2,2)]);        // (range, slice)
    CHECK_VIEW_RMC(array[ndarray::view()(0,2)]);          // (range, range)
    CHECK_VIEW_RMC(array[ndarray::view()()]);             // (range, full)
    CHECK_VIEW_RMC(array[ndarray::view()(1)]);            // (range, scalar)

    CHECK_VIEW_RMC(array[ndarray::view(2)(0,2,2)]);       // (scalar, slice)
    CHECK_VIEW_RMC(array[ndarray::view(2)(0,2)]);         // (scalar, range)
    CHECK_VIEW_RMC(array[ndarray::view(2)()]);            // (scalar, full)

}

BOOST_AUTO_TEST_CASE(view2) {

    ndarray::Array<float,2,2> parent = ndarray::allocate(10,8);
    ndarray::Array<float,2,0> array = parent[ndarray::view(0,10,2)(0,8,2)];

    CHECK_VIEW_RMC(array[ndarray::view(1,3,2)(0,2,2)]);   // (slice, slice)
    CHECK_VIEW_RMC(array[ndarray::view(1,3,2)(0,2)]);     // (slice, range)
    CHECK_VIEW_RMC(array[ndarray::view(1,3,2)()]);        // (slice, full)
    CHECK_VIEW_RMC(array[ndarray::view(1,3,2)(1)]);       // (slice, scalar)

    CHECK_VIEW_RMC(array[ndarray::view(1,3)(0,2,2)]);     // (range, slice)
    CHECK_VIEW_RMC(array[ndarray::view(1,3)(0,2)]);       // (range, range)
    CHECK_VIEW_RMC(array[ndarray::view(1,3)()]);          // (range, full)
    CHECK_VIEW_RMC(array[ndarray::view(1,3)(1)]);         // (range, scalar)

    CHECK_VIEW_RMC(array[ndarray::view()(0,2,2)]);        // (range, slice)
    CHECK_VIEW_RMC(array[ndarray::view()(0,2)]);          // (range, range)
    CHECK_VIEW_RMC(array[ndarray::view()()]);             // (range, full)
    CHECK_VIEW_RMC(array[ndarray::view()(1)]);            // (range, scalar)

    CHECK_VIEW_RMC(array[ndarray::view(2)(0,2,2)]);       // (scalar, slice)
    CHECK_VIEW_RMC(array[ndarray::view(2)(0,2)]);         // (scalar, range)
    CHECK_VIEW_RMC(array[ndarray::view(2)()]);            // (scalar, full)

}

BOOST_AUTO_TEST_CASE(view2m2) {

    ndarray::Array<float,2,2> parent = ndarray::allocate(4,5);
    ndarray::Array<float,2,-2> array = parent.transpose();

    CHECK_VIEW_RMC(array[ndarray::view(1,3,2)(0,2,2)]);   // (slice, slice)
    CHECK_VIEW_RMC(array[ndarray::view(1,3,2)(0,2)]);     // (slice, range)
    CHECK_VIEW_RMC(array[ndarray::view(1,3,2)()]);        // (slice, full)
    CHECK_VIEW_RMC(array[ndarray::view(1,3,2)(1)]);       // (slice, scalar)

    CHECK_VIEW_RMC(array[ndarray::view(1,3)(0,2,2)]);     // (range, slice)
    CHECK_VIEW_RMC(array[ndarray::view(1,3)(0,2)]);       // (range, range)
    CHECK_VIEW_RMC(array[ndarray::view(1,3)()]);          // (range, full)
    CHECK_VIEW_RMC(array[ndarray::view(1,3)(1)]);         // (range, scalar)

    CHECK_VIEW_RMC(array[ndarray::view()(0,2,2)]);        // (range, slice)
    CHECK_VIEW_RMC(array[ndarray::view()(0,2)]);          // (range, range)
    CHECK_VIEW_RMC(array[ndarray::view()()]);             // (range, full)
    CHECK_VIEW_RMC(array[ndarray::view()(1)]);            // (range, scalar)

    CHECK_VIEW_RMC(array[ndarray::view(2)(0,2,2)]);       // (scalar, slice)
    CHECK_VIEW_RMC(array[ndarray::view(2)(0,2)]);         // (scalar, range)
    CHECK_VIEW_RMC(array[ndarray::view(2)()]);            // (scalar, full)

}

BOOST_AUTO_TEST_CASE(view2m1) {

    ndarray::Array<float,2,2> parent1 = ndarray::allocate(8,5);
    ndarray::Array<float,2,-2> parent2 = parent1.transpose();
    ndarray::Array<float,2,-1> array = parent2[ndarray::view()(0,8,2)];

    CHECK_VIEW_RMC(array[ndarray::view(1,3,2)(0,2,2)]);   // (slice, slice)
    CHECK_VIEW_RMC(array[ndarray::view(1,3,2)(0,2)]);     // (slice, range)
    CHECK_VIEW_RMC(array[ndarray::view(1,3,2)()]);        // (slice, full)
    CHECK_VIEW_RMC(array[ndarray::view(1,3,2)(1)]);       // (slice, scalar)

    CHECK_VIEW_RMC(array[ndarray::view(1,3)(0,2,2)]);     // (range, slice)
    CHECK_VIEW_RMC(array[ndarray::view(1,3)(0,2)]);       // (range, range)
    CHECK_VIEW_RMC(array[ndarray::view(1,3)()]);          // (range, full)
    CHECK_VIEW_RMC(array[ndarray::view(1,3)(1)]);         // (range, scalar)

    CHECK_VIEW_RMC(array[ndarray::view()(0,2,2)]);        // (range, slice)
    CHECK_VIEW_RMC(array[ndarray::view()(0,2)]);          // (range, range)
    CHECK_VIEW_RMC(array[ndarray::view()()]);             // (range, full)
    CHECK_VIEW_RMC(array[ndarray::view()(1)]);            // (range, scalar)

    CHECK_VIEW_RMC(array[ndarray::view(2)(0,2,2)]);       // (scalar, slice)
    CHECK_VIEW_RMC(array[ndarray::view(2)(0,2)]);         // (scalar, range)
    CHECK_VIEW_RMC(array[ndarray::view(2)()]);            // (scalar, full)

}

BOOST_AUTO_TEST_CASE(view3p3) {

    ndarray::Array<float,3,3> array = ndarray::allocate(6,5,4);

    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)(1,3,2)(0,2,2)]);   // (slice, slice, slice)
    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)(1,3,2)(0,2)]);     // (slice, slice, range)
    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)(1,3,2)()]);        // (slice, slice, full)
    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)(1,3,2)(1)]);       // (slice, slice, scalar)

    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)(1,3)(0,2,2)]);     // (slice, range, slice)
    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)(1,3)(0,2)]);       // (slice, range, range)
    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)(1,3)()]);          // (slice, range, full)
    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)(1,3)(1)]);         // (slice, range, scalar)

    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)()(0,2,2)]);        // (slice, range, slice)
    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)()(0,2)]);          // (slice, range, range)
    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)()()]);             // (slice, range, full)
    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)()(1)]);            // (slice, range, scalar)

    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)(2)(0,2,2)]);       // (slice, scalar, slice)
    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)(2)(0,2)]);         // (slice, scalar, range)
    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)(2)()]);            // (slice, scalar, full)

    CHECK_VIEW_RMC(array[ndarray::view(1,5)(1,3,2)(0,2,2)]);     // (range, slice, slice)
    CHECK_VIEW_RMC(array[ndarray::view(1,5)(1,3,2)(0,2)]);       // (range, slice, range)
    CHECK_VIEW_RMC(array[ndarray::view(1,5)(1,3,2)()]);          // (range, slice, full)
    CHECK_VIEW_RMC(array[ndarray::view(1,5)(1,3,2)(1)]);         // (range, slice, scalar)

    CHECK_VIEW_RMC(array[ndarray::view(1,5)(1,3)(0,2,2)]);       // (range, range, slice)
    CHECK_VIEW_RMC(array[ndarray::view(1,5)(1,3)(0,2)]);         // (range, range, range)
    CHECK_VIEW_RMC(array[ndarray::view(1,5)(1,3)()]);            // (range, range, full)
    CHECK_VIEW_RMC(array[ndarray::view(1,5)(1,3)(1)]);           // (range, range, scalar)

    CHECK_VIEW_RMC(array[ndarray::view(1,5)()(0,2,2)]);          // (range, range, slice)
    CHECK_VIEW_RMC(array[ndarray::view(1,5)()(0,2)]);            // (range, range, range)
    CHECK_VIEW_RMC(array[ndarray::view(1,5)()()]);               // (range, range, full)
    CHECK_VIEW_RMC(array[ndarray::view(1,5)()(1)]);              // (range, range, scalar)

    CHECK_VIEW_RMC(array[ndarray::view(1,5)(2)(0,2,2)]);         // (range, scalar, slice)
    CHECK_VIEW_RMC(array[ndarray::view(1,5)(2)(0,2)]);           // (range, scalar, range)
    CHECK_VIEW_RMC(array[ndarray::view(1,5)(2)()]);              // (range, scalar, full)

    CHECK_VIEW_RMC(array[ndarray::view()(1,3,2)(0,2,2)]);        // (full, slice, slice)
    CHECK_VIEW_RMC(array[ndarray::view()(1,3,2)(0,2)]);          // (full, slice, range)
    CHECK_VIEW_RMC(array[ndarray::view()(1,3,2)()]);             // (full, slice, full)
    CHECK_VIEW_RMC(array[ndarray::view()(1,3,2)(1)]);            // (full, slice, scalar)

    CHECK_VIEW_RMC(array[ndarray::view()(1,3)(0,2,2)]);          // (full, range, slice)
    CHECK_VIEW_RMC(array[ndarray::view()(1,3)(0,2)]);            // (full, range, range)
    CHECK_VIEW_RMC(array[ndarray::view()(1,3)()]);               // (full, range, full)
    CHECK_VIEW_RMC(array[ndarray::view()(1,3)(1)]);              // (full, range, scalar)

    CHECK_VIEW_RMC(array[ndarray::view()()(0,2,2)]);             // (full, range, slice)
    CHECK_VIEW_RMC(array[ndarray::view()()(0,2)]);               // (full, range, range)
    CHECK_VIEW_RMC(array[ndarray::view()()()]);                  // (full, range, full)
    CHECK_VIEW_RMC(array[ndarray::view()()(1)]);                 // (full, range, scalar)

    CHECK_VIEW_RMC(array[ndarray::view()(2)(0,2,2)]);            // (full, scalar, slice)
    CHECK_VIEW_RMC(array[ndarray::view()(2)(0,2)]);              // (full, scalar, range)
    CHECK_VIEW_RMC(array[ndarray::view()(2)()]);                 // (full, scalar, full)

    CHECK_VIEW_RMC(array[ndarray::view(3)(1,3,2)(0,2,2)]);        // (scalar, slice, slice)
    CHECK_VIEW_RMC(array[ndarray::view(3)(1,3,2)(0,2)]);          // (scalar, slice, range)
    CHECK_VIEW_RMC(array[ndarray::view(3)(1,3,2)()]);             // (scalar, slice, full)
    CHECK_VIEW_RMC(array[ndarray::view(3)(1,3,2)(1)]);            // (scalar, slice, scalar)

    CHECK_VIEW_RMC(array[ndarray::view(3)(1,3)(0,2,2)]);          // (scalar, range, slice)
    CHECK_VIEW_RMC(array[ndarray::view(3)(1,3)(0,2)]);            // (scalar, range, range)
    CHECK_VIEW_RMC(array[ndarray::view(3)(1,3)()]);               // (scalar, range, full)
    CHECK_VIEW_RMC(array[ndarray::view(3)(1,3)(1)]);              // (scalar, range, scalar)

    CHECK_VIEW_RMC(array[ndarray::view(3)()(0,2,2)]);             // (scalar, range, slice)
    CHECK_VIEW_RMC(array[ndarray::view(3)()(0,2)]);               // (scalar, range, range)
    CHECK_VIEW_RMC(array[ndarray::view(3)()()]);                  // (scalar, range, full)
    CHECK_VIEW_RMC(array[ndarray::view(3)()(1)]);                 // (scalar, range, scalar)

    CHECK_VIEW_RMC(array[ndarray::view(3)(2)(0,2,2)]);            // (scalar, scalar, slice)
    CHECK_VIEW_RMC(array[ndarray::view(3)(2)(0,2)]);              // (scalar, scalar, range)
    CHECK_VIEW_RMC(array[ndarray::view(3)(2)()]);                 // (scalar, scalar, full)

}

BOOST_AUTO_TEST_CASE(view3p2) {

    ndarray::Array<float,3,3> parent = ndarray::allocate(12,5,4);
    ndarray::Array<float,3,2> array = parent[ndarray::view(0,12,2)()()];

    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)(1,3,2)(0,2,2)]);   // (slice, slice, slice)
    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)(1,3,2)(0,2)]);     // (slice, slice, range)
    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)(1,3,2)()]);        // (slice, slice, full)
    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)(1,3,2)(1)]);       // (slice, slice, scalar)

    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)(1,3)(0,2,2)]);     // (slice, range, slice)
    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)(1,3)(0,2)]);       // (slice, range, range)
    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)(1,3)()]);          // (slice, range, full)
    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)(1,3)(1)]);         // (slice, range, scalar)

    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)()(0,2,2)]);        // (slice, range, slice)
    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)()(0,2)]);          // (slice, range, range)
    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)()()]);             // (slice, range, full)
    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)()(1)]);            // (slice, range, scalar)

    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)(2)(0,2,2)]);       // (slice, scalar, slice)
    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)(2)(0,2)]);         // (slice, scalar, range)
    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)(2)()]);            // (slice, scalar, full)

    CHECK_VIEW_RMC(array[ndarray::view(1,5)(1,3,2)(0,2,2)]);     // (range, slice, slice)
    CHECK_VIEW_RMC(array[ndarray::view(1,5)(1,3,2)(0,2)]);       // (range, slice, range)
    CHECK_VIEW_RMC(array[ndarray::view(1,5)(1,3,2)()]);          // (range, slice, full)
    CHECK_VIEW_RMC(array[ndarray::view(1,5)(1,3,2)(1)]);         // (range, slice, scalar)

    CHECK_VIEW_RMC(array[ndarray::view(1,5)(1,3)(0,2,2)]);       // (range, range, slice)
    CHECK_VIEW_RMC(array[ndarray::view(1,5)(1,3)(0,2)]);         // (range, range, range)
    CHECK_VIEW_RMC(array[ndarray::view(1,5)(1,3)()]);            // (range, range, full)
    CHECK_VIEW_RMC(array[ndarray::view(1,5)(1,3)(1)]);           // (range, range, scalar)

    CHECK_VIEW_RMC(array[ndarray::view(1,5)()(0,2,2)]);          // (range, range, slice)
    CHECK_VIEW_RMC(array[ndarray::view(1,5)()(0,2)]);            // (range, range, range)
    CHECK_VIEW_RMC(array[ndarray::view(1,5)()()]);               // (range, range, full)
    CHECK_VIEW_RMC(array[ndarray::view(1,5)()(1)]);              // (range, range, scalar)

    CHECK_VIEW_RMC(array[ndarray::view(1,5)(2)(0,2,2)]);         // (range, scalar, slice)
    CHECK_VIEW_RMC(array[ndarray::view(1,5)(2)(0,2)]);           // (range, scalar, range)
    CHECK_VIEW_RMC(array[ndarray::view(1,5)(2)()]);              // (range, scalar, full)

    CHECK_VIEW_RMC(array[ndarray::view()(1,3,2)(0,2,2)]);        // (full, slice, slice)
    CHECK_VIEW_RMC(array[ndarray::view()(1,3,2)(0,2)]);          // (full, slice, range)
    CHECK_VIEW_RMC(array[ndarray::view()(1,3,2)()]);             // (full, slice, full)
    CHECK_VIEW_RMC(array[ndarray::view()(1,3,2)(1)]);            // (full, slice, scalar)

    CHECK_VIEW_RMC(array[ndarray::view()(1,3)(0,2,2)]);          // (full, range, slice)
    CHECK_VIEW_RMC(array[ndarray::view()(1,3)(0,2)]);            // (full, range, range)
    CHECK_VIEW_RMC(array[ndarray::view()(1,3)()]);               // (full, range, full)
    CHECK_VIEW_RMC(array[ndarray::view()(1,3)(1)]);              // (full, range, scalar)

    CHECK_VIEW_RMC(array[ndarray::view()()(0,2,2)]);             // (full, range, slice)
    CHECK_VIEW_RMC(array[ndarray::view()()(0,2)]);               // (full, range, range)
    CHECK_VIEW_RMC(array[ndarray::view()()()]);                  // (full, range, full)
    CHECK_VIEW_RMC(array[ndarray::view()()(1)]);                 // (full, range, scalar)

    CHECK_VIEW_RMC(array[ndarray::view()(2)(0,2,2)]);            // (full, scalar, slice)
    CHECK_VIEW_RMC(array[ndarray::view()(2)(0,2)]);              // (full, scalar, range)
    CHECK_VIEW_RMC(array[ndarray::view()(2)()]);                 // (full, scalar, full)

    CHECK_VIEW_RMC(array[ndarray::view(3)(1,3,2)(0,2,2)]);        // (scalar, slice, slice)
    CHECK_VIEW_RMC(array[ndarray::view(3)(1,3,2)(0,2)]);          // (scalar, slice, range)
    CHECK_VIEW_RMC(array[ndarray::view(3)(1,3,2)()]);             // (scalar, slice, full)
    CHECK_VIEW_RMC(array[ndarray::view(3)(1,3,2)(1)]);            // (scalar, slice, scalar)

    CHECK_VIEW_RMC(array[ndarray::view(3)(1,3)(0,2,2)]);          // (scalar, range, slice)
    CHECK_VIEW_RMC(array[ndarray::view(3)(1,3)(0,2)]);            // (scalar, range, range)
    CHECK_VIEW_RMC(array[ndarray::view(3)(1,3)()]);               // (scalar, range, full)
    CHECK_VIEW_RMC(array[ndarray::view(3)(1,3)(1)]);              // (scalar, range, scalar)

    CHECK_VIEW_RMC(array[ndarray::view(3)()(0,2,2)]);             // (scalar, range, slice)
    CHECK_VIEW_RMC(array[ndarray::view(3)()(0,2)]);               // (scalar, range, range)
    CHECK_VIEW_RMC(array[ndarray::view(3)()()]);                  // (scalar, range, full)
    CHECK_VIEW_RMC(array[ndarray::view(3)()(1)]);                 // (scalar, range, scalar)

    CHECK_VIEW_RMC(array[ndarray::view(3)(2)(0,2,2)]);            // (scalar, scalar, slice)
    CHECK_VIEW_RMC(array[ndarray::view(3)(2)(0,2)]);              // (scalar, scalar, range)
    CHECK_VIEW_RMC(array[ndarray::view(3)(2)()]);                 // (scalar, scalar, full)

}

BOOST_AUTO_TEST_CASE(view3p1) {

    ndarray::Array<float,3,3> parent = ndarray::allocate(12,10,4);
    ndarray::Array<float,3,1> array = parent[ndarray::view(0,12,2)(0,10,2)()];

    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)(1,3,2)(0,2,2)]);   // (slice, slice, slice)
    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)(1,3,2)(0,2)]);     // (slice, slice, range)
    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)(1,3,2)()]);        // (slice, slice, full)
    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)(1,3,2)(1)]);       // (slice, slice, scalar)

    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)(1,3)(0,2,2)]);     // (slice, range, slice)
    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)(1,3)(0,2)]);       // (slice, range, range)
    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)(1,3)()]);          // (slice, range, full)
    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)(1,3)(1)]);         // (slice, range, scalar)

    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)()(0,2,2)]);        // (slice, range, slice)
    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)()(0,2)]);          // (slice, range, range)
    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)()()]);             // (slice, range, full)
    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)()(1)]);            // (slice, range, scalar)

    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)(2)(0,2,2)]);       // (slice, scalar, slice)
    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)(2)(0,2)]);         // (slice, scalar, range)
    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)(2)()]);            // (slice, scalar, full)

    CHECK_VIEW_RMC(array[ndarray::view(1,5)(1,3,2)(0,2,2)]);     // (range, slice, slice)
    CHECK_VIEW_RMC(array[ndarray::view(1,5)(1,3,2)(0,2)]);       // (range, slice, range)
    CHECK_VIEW_RMC(array[ndarray::view(1,5)(1,3,2)()]);          // (range, slice, full)
    CHECK_VIEW_RMC(array[ndarray::view(1,5)(1,3,2)(1)]);         // (range, slice, scalar)

    CHECK_VIEW_RMC(array[ndarray::view(1,5)(1,3)(0,2,2)]);       // (range, range, slice)
    CHECK_VIEW_RMC(array[ndarray::view(1,5)(1,3)(0,2)]);         // (range, range, range)
    CHECK_VIEW_RMC(array[ndarray::view(1,5)(1,3)()]);            // (range, range, full)
    CHECK_VIEW_RMC(array[ndarray::view(1,5)(1,3)(1)]);           // (range, range, scalar)

    CHECK_VIEW_RMC(array[ndarray::view(1,5)()(0,2,2)]);          // (range, range, slice)
    CHECK_VIEW_RMC(array[ndarray::view(1,5)()(0,2)]);            // (range, range, range)
    CHECK_VIEW_RMC(array[ndarray::view(1,5)()()]);               // (range, range, full)
    CHECK_VIEW_RMC(array[ndarray::view(1,5)()(1)]);              // (range, range, scalar)

    CHECK_VIEW_RMC(array[ndarray::view(1,5)(2)(0,2,2)]);         // (range, scalar, slice)
    CHECK_VIEW_RMC(array[ndarray::view(1,5)(2)(0,2)]);           // (range, scalar, range)
    CHECK_VIEW_RMC(array[ndarray::view(1,5)(2)()]);              // (range, scalar, full)

    CHECK_VIEW_RMC(array[ndarray::view()(1,3,2)(0,2,2)]);        // (full, slice, slice)
    CHECK_VIEW_RMC(array[ndarray::view()(1,3,2)(0,2)]);          // (full, slice, range)
    CHECK_VIEW_RMC(array[ndarray::view()(1,3,2)()]);             // (full, slice, full)
    CHECK_VIEW_RMC(array[ndarray::view()(1,3,2)(1)]);            // (full, slice, scalar)

    CHECK_VIEW_RMC(array[ndarray::view()(1,3)(0,2,2)]);          // (full, range, slice)
    CHECK_VIEW_RMC(array[ndarray::view()(1,3)(0,2)]);            // (full, range, range)
    CHECK_VIEW_RMC(array[ndarray::view()(1,3)()]);               // (full, range, full)
    CHECK_VIEW_RMC(array[ndarray::view()(1,3)(1)]);              // (full, range, scalar)

    CHECK_VIEW_RMC(array[ndarray::view()()(0,2,2)]);             // (full, range, slice)
    CHECK_VIEW_RMC(array[ndarray::view()()(0,2)]);               // (full, range, range)
    CHECK_VIEW_RMC(array[ndarray::view()()()]);                  // (full, range, full)
    CHECK_VIEW_RMC(array[ndarray::view()()(1)]);                 // (full, range, scalar)

    CHECK_VIEW_RMC(array[ndarray::view()(2)(0,2,2)]);            // (full, scalar, slice)
    CHECK_VIEW_RMC(array[ndarray::view()(2)(0,2)]);              // (full, scalar, range)
    CHECK_VIEW_RMC(array[ndarray::view()(2)()]);                 // (full, scalar, full)

    CHECK_VIEW_RMC(array[ndarray::view(3)(1,3,2)(0,2,2)]);        // (scalar, slice, slice)
    CHECK_VIEW_RMC(array[ndarray::view(3)(1,3,2)(0,2)]);          // (scalar, slice, range)
    CHECK_VIEW_RMC(array[ndarray::view(3)(1,3,2)()]);             // (scalar, slice, full)
    CHECK_VIEW_RMC(array[ndarray::view(3)(1,3,2)(1)]);            // (scalar, slice, scalar)

    CHECK_VIEW_RMC(array[ndarray::view(3)(1,3)(0,2,2)]);          // (scalar, range, slice)
    CHECK_VIEW_RMC(array[ndarray::view(3)(1,3)(0,2)]);            // (scalar, range, range)
    CHECK_VIEW_RMC(array[ndarray::view(3)(1,3)()]);               // (scalar, range, full)
    CHECK_VIEW_RMC(array[ndarray::view(3)(1,3)(1)]);              // (scalar, range, scalar)

    CHECK_VIEW_RMC(array[ndarray::view(3)()(0,2,2)]);             // (scalar, range, slice)
    CHECK_VIEW_RMC(array[ndarray::view(3)()(0,2)]);               // (scalar, range, range)
    CHECK_VIEW_RMC(array[ndarray::view(3)()()]);                  // (scalar, range, full)
    CHECK_VIEW_RMC(array[ndarray::view(3)()(1)]);                 // (scalar, range, scalar)

    CHECK_VIEW_RMC(array[ndarray::view(3)(2)(0,2,2)]);            // (scalar, scalar, slice)
    CHECK_VIEW_RMC(array[ndarray::view(3)(2)(0,2)]);              // (scalar, scalar, range)
    CHECK_VIEW_RMC(array[ndarray::view(3)(2)()]);                 // (scalar, scalar, full)

}

BOOST_AUTO_TEST_CASE(view3p0) {

    ndarray::Array<float,3,3> parent = ndarray::allocate(12,10,8);
    ndarray::Array<float,3,0> array = parent[ndarray::view(0,12,2)(0,10,2)(0,8,2)];

    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)(1,3,2)(0,2,2)]);   // (slice, slice, slice)
    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)(1,3,2)(0,2)]);     // (slice, slice, range)
    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)(1,3,2)()]);        // (slice, slice, full)
    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)(1,3,2)(1)]);       // (slice, slice, scalar)

    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)(1,3)(0,2,2)]);     // (slice, range, slice)
    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)(1,3)(0,2)]);       // (slice, range, range)
    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)(1,3)()]);          // (slice, range, full)
    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)(1,3)(1)]);         // (slice, range, scalar)

    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)()(0,2,2)]);        // (slice, range, slice)
    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)()(0,2)]);          // (slice, range, range)
    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)()()]);             // (slice, range, full)
    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)()(1)]);            // (slice, range, scalar)

    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)(2)(0,2,2)]);       // (slice, scalar, slice)
    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)(2)(0,2)]);         // (slice, scalar, range)
    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)(2)()]);            // (slice, scalar, full)

    CHECK_VIEW_RMC(array[ndarray::view(1,5)(1,3,2)(0,2,2)]);     // (range, slice, slice)
    CHECK_VIEW_RMC(array[ndarray::view(1,5)(1,3,2)(0,2)]);       // (range, slice, range)
    CHECK_VIEW_RMC(array[ndarray::view(1,5)(1,3,2)()]);          // (range, slice, full)
    CHECK_VIEW_RMC(array[ndarray::view(1,5)(1,3,2)(1)]);         // (range, slice, scalar)

    CHECK_VIEW_RMC(array[ndarray::view(1,5)(1,3)(0,2,2)]);       // (range, range, slice)
    CHECK_VIEW_RMC(array[ndarray::view(1,5)(1,3)(0,2)]);         // (range, range, range)
    CHECK_VIEW_RMC(array[ndarray::view(1,5)(1,3)()]);            // (range, range, full)
    CHECK_VIEW_RMC(array[ndarray::view(1,5)(1,3)(1)]);           // (range, range, scalar)

    CHECK_VIEW_RMC(array[ndarray::view(1,5)()(0,2,2)]);          // (range, range, slice)
    CHECK_VIEW_RMC(array[ndarray::view(1,5)()(0,2)]);            // (range, range, range)
    CHECK_VIEW_RMC(array[ndarray::view(1,5)()()]);               // (range, range, full)
    CHECK_VIEW_RMC(array[ndarray::view(1,5)()(1)]);              // (range, range, scalar)

    CHECK_VIEW_RMC(array[ndarray::view(1,5)(2)(0,2,2)]);         // (range, scalar, slice)
    CHECK_VIEW_RMC(array[ndarray::view(1,5)(2)(0,2)]);           // (range, scalar, range)
    CHECK_VIEW_RMC(array[ndarray::view(1,5)(2)()]);              // (range, scalar, full)

    CHECK_VIEW_RMC(array[ndarray::view()(1,3,2)(0,2,2)]);        // (full, slice, slice)
    CHECK_VIEW_RMC(array[ndarray::view()(1,3,2)(0,2)]);          // (full, slice, range)
    CHECK_VIEW_RMC(array[ndarray::view()(1,3,2)()]);             // (full, slice, full)
    CHECK_VIEW_RMC(array[ndarray::view()(1,3,2)(1)]);            // (full, slice, scalar)

    CHECK_VIEW_RMC(array[ndarray::view()(1,3)(0,2,2)]);          // (full, range, slice)
    CHECK_VIEW_RMC(array[ndarray::view()(1,3)(0,2)]);            // (full, range, range)
    CHECK_VIEW_RMC(array[ndarray::view()(1,3)()]);               // (full, range, full)
    CHECK_VIEW_RMC(array[ndarray::view()(1,3)(1)]);              // (full, range, scalar)

    CHECK_VIEW_RMC(array[ndarray::view()()(0,2,2)]);             // (full, range, slice)
    CHECK_VIEW_RMC(array[ndarray::view()()(0,2)]);               // (full, range, range)
    CHECK_VIEW_RMC(array[ndarray::view()()()]);                  // (full, range, full)
    CHECK_VIEW_RMC(array[ndarray::view()()(1)]);                 // (full, range, scalar)

    CHECK_VIEW_RMC(array[ndarray::view()(2)(0,2,2)]);            // (full, scalar, slice)
    CHECK_VIEW_RMC(array[ndarray::view()(2)(0,2)]);              // (full, scalar, range)
    CHECK_VIEW_RMC(array[ndarray::view()(2)()]);                 // (full, scalar, full)

    CHECK_VIEW_RMC(array[ndarray::view(3)(1,3,2)(0,2,2)]);        // (scalar, slice, slice)
    CHECK_VIEW_RMC(array[ndarray::view(3)(1,3,2)(0,2)]);          // (scalar, slice, range)
    CHECK_VIEW_RMC(array[ndarray::view(3)(1,3,2)()]);             // (scalar, slice, full)
    CHECK_VIEW_RMC(array[ndarray::view(3)(1,3,2)(1)]);            // (scalar, slice, scalar)

    CHECK_VIEW_RMC(array[ndarray::view(3)(1,3)(0,2,2)]);          // (scalar, range, slice)
    CHECK_VIEW_RMC(array[ndarray::view(3)(1,3)(0,2)]);            // (scalar, range, range)
    CHECK_VIEW_RMC(array[ndarray::view(3)(1,3)()]);               // (scalar, range, full)
    CHECK_VIEW_RMC(array[ndarray::view(3)(1,3)(1)]);              // (scalar, range, scalar)

    CHECK_VIEW_RMC(array[ndarray::view(3)()(0,2,2)]);             // (scalar, range, slice)
    CHECK_VIEW_RMC(array[ndarray::view(3)()(0,2)]);               // (scalar, range, range)
    CHECK_VIEW_RMC(array[ndarray::view(3)()()]);                  // (scalar, range, full)
    CHECK_VIEW_RMC(array[ndarray::view(3)()(1)]);                 // (scalar, range, scalar)

    CHECK_VIEW_RMC(array[ndarray::view(3)(2)(0,2,2)]);            // (scalar, scalar, slice)
    CHECK_VIEW_RMC(array[ndarray::view(3)(2)(0,2)]);              // (scalar, scalar, range)
    CHECK_VIEW_RMC(array[ndarray::view(3)(2)()]);                 // (scalar, scalar, full)

}

BOOST_AUTO_TEST_CASE(view3m2) {

    ndarray::Array<float,3,3> parent1 = ndarray::allocate(8,5,6);
    ndarray::Array<float,3,-3> parent2 = parent1.transpose();
    ndarray::Array<float,3,-2> array = parent2[ndarray::view()()(0,8,2)];

    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)(1,3,2)(0,2,2)]);   // (slice, slice, slice)
    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)(1,3,2)(0,2)]);     // (slice, slice, range)
    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)(1,3,2)()]);        // (slice, slice, full)
    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)(1,3,2)(1)]);       // (slice, slice, scalar)

    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)(1,3)(0,2,2)]);     // (slice, range, slice)
    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)(1,3)(0,2)]);       // (slice, range, range)
    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)(1,3)()]);          // (slice, range, full)
    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)(1,3)(1)]);         // (slice, range, scalar)

    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)()(0,2,2)]);        // (slice, range, slice)
    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)()(0,2)]);          // (slice, range, range)
    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)()()]);             // (slice, range, full)
    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)()(1)]);            // (slice, range, scalar)

    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)(2)(0,2,2)]);       // (slice, scalar, slice)
    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)(2)(0,2)]);         // (slice, scalar, range)
    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)(2)()]);            // (slice, scalar, full)

    CHECK_VIEW_RMC(array[ndarray::view(1,5)(1,3,2)(0,2,2)]);     // (range, slice, slice)
    CHECK_VIEW_RMC(array[ndarray::view(1,5)(1,3,2)(0,2)]);       // (range, slice, range)
    CHECK_VIEW_RMC(array[ndarray::view(1,5)(1,3,2)()]);          // (range, slice, full)
    CHECK_VIEW_RMC(array[ndarray::view(1,5)(1,3,2)(1)]);         // (range, slice, scalar)

    CHECK_VIEW_RMC(array[ndarray::view(1,5)(1,3)(0,2,2)]);       // (range, range, slice)
    CHECK_VIEW_RMC(array[ndarray::view(1,5)(1,3)(0,2)]);         // (range, range, range)
    CHECK_VIEW_RMC(array[ndarray::view(1,5)(1,3)()]);            // (range, range, full)
    CHECK_VIEW_RMC(array[ndarray::view(1,5)(1,3)(1)]);           // (range, range, scalar)

    CHECK_VIEW_RMC(array[ndarray::view(1,5)()(0,2,2)]);          // (range, range, slice)
    CHECK_VIEW_RMC(array[ndarray::view(1,5)()(0,2)]);            // (range, range, range)
    CHECK_VIEW_RMC(array[ndarray::view(1,5)()()]);               // (range, range, full)
    CHECK_VIEW_RMC(array[ndarray::view(1,5)()(1)]);              // (range, range, scalar)

    CHECK_VIEW_RMC(array[ndarray::view(1,5)(2)(0,2,2)]);         // (range, scalar, slice)
    CHECK_VIEW_RMC(array[ndarray::view(1,5)(2)(0,2)]);           // (range, scalar, range)
    CHECK_VIEW_RMC(array[ndarray::view(1,5)(2)()]);              // (range, scalar, full)

    CHECK_VIEW_RMC(array[ndarray::view()(1,3,2)(0,2,2)]);        // (full, slice, slice)
    CHECK_VIEW_RMC(array[ndarray::view()(1,3,2)(0,2)]);          // (full, slice, range)
    CHECK_VIEW_RMC(array[ndarray::view()(1,3,2)()]);             // (full, slice, full)
    CHECK_VIEW_RMC(array[ndarray::view()(1,3,2)(1)]);            // (full, slice, scalar)

    CHECK_VIEW_RMC(array[ndarray::view()(1,3)(0,2,2)]);          // (full, range, slice)
    CHECK_VIEW_RMC(array[ndarray::view()(1,3)(0,2)]);            // (full, range, range)
    CHECK_VIEW_RMC(array[ndarray::view()(1,3)()]);               // (full, range, full)
    CHECK_VIEW_RMC(array[ndarray::view()(1,3)(1)]);              // (full, range, scalar)

    CHECK_VIEW_RMC(array[ndarray::view()()(0,2,2)]);             // (full, range, slice)
    CHECK_VIEW_RMC(array[ndarray::view()()(0,2)]);               // (full, range, range)
    CHECK_VIEW_RMC(array[ndarray::view()()()]);                  // (full, range, full)
    CHECK_VIEW_RMC(array[ndarray::view()()(1)]);                 // (full, range, scalar)

    CHECK_VIEW_RMC(array[ndarray::view()(2)(0,2,2)]);            // (full, scalar, slice)
    CHECK_VIEW_RMC(array[ndarray::view()(2)(0,2)]);              // (full, scalar, range)
    CHECK_VIEW_RMC(array[ndarray::view()(2)()]);                 // (full, scalar, full)

    CHECK_VIEW_RMC(array[ndarray::view(3)(1,3,2)(0,2,2)]);        // (scalar, slice, slice)
    CHECK_VIEW_RMC(array[ndarray::view(3)(1,3,2)(0,2)]);          // (scalar, slice, range)
    CHECK_VIEW_RMC(array[ndarray::view(3)(1,3,2)()]);             // (scalar, slice, full)
    CHECK_VIEW_RMC(array[ndarray::view(3)(1,3,2)(1)]);            // (scalar, slice, scalar)

    CHECK_VIEW_RMC(array[ndarray::view(3)(1,3)(0,2,2)]);          // (scalar, range, slice)
    CHECK_VIEW_RMC(array[ndarray::view(3)(1,3)(0,2)]);            // (scalar, range, range)
    CHECK_VIEW_RMC(array[ndarray::view(3)(1,3)()]);               // (scalar, range, full)
    CHECK_VIEW_RMC(array[ndarray::view(3)(1,3)(1)]);              // (scalar, range, scalar)

    CHECK_VIEW_RMC(array[ndarray::view(3)()(0,2,2)]);             // (scalar, range, slice)
    CHECK_VIEW_RMC(array[ndarray::view(3)()(0,2)]);               // (scalar, range, range)
    CHECK_VIEW_RMC(array[ndarray::view(3)()()]);                  // (scalar, range, full)
    CHECK_VIEW_RMC(array[ndarray::view(3)()(1)]);                 // (scalar, range, scalar)

    CHECK_VIEW_RMC(array[ndarray::view(3)(2)(0,2,2)]);            // (scalar, scalar, slice)
    CHECK_VIEW_RMC(array[ndarray::view(3)(2)(0,2)]);              // (scalar, scalar, range)
    CHECK_VIEW_RMC(array[ndarray::view(3)(2)()]);                 // (scalar, scalar, full)

}

BOOST_AUTO_TEST_CASE(view3m1) {

    ndarray::Array<float,3,3> parent1 = ndarray::allocate(8,10,6);
    ndarray::Array<float,3,-3> parent2 = parent1.transpose();
    ndarray::Array<float,3,-1> array = parent2[ndarray::view()(0,10,2)(0,8,2)];

    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)(1,3,2)(0,2,2)]);   // (slice, slice, slice)
    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)(1,3,2)(0,2)]);     // (slice, slice, range)
    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)(1,3,2)()]);        // (slice, slice, full)
    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)(1,3,2)(1)]);       // (slice, slice, scalar)

    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)(1,3)(0,2,2)]);     // (slice, range, slice)
    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)(1,3)(0,2)]);       // (slice, range, range)
    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)(1,3)()]);          // (slice, range, full)
    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)(1,3)(1)]);         // (slice, range, scalar)

    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)()(0,2,2)]);        // (slice, range, slice)
    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)()(0,2)]);          // (slice, range, range)
    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)()()]);             // (slice, range, full)
    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)()(1)]);            // (slice, range, scalar)

    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)(2)(0,2,2)]);       // (slice, scalar, slice)
    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)(2)(0,2)]);         // (slice, scalar, range)
    CHECK_VIEW_RMC(array[ndarray::view(2,6,2)(2)()]);            // (slice, scalar, full)

    CHECK_VIEW_RMC(array[ndarray::view(1,5)(1,3,2)(0,2,2)]);     // (range, slice, slice)
    CHECK_VIEW_RMC(array[ndarray::view(1,5)(1,3,2)(0,2)]);       // (range, slice, range)
    CHECK_VIEW_RMC(array[ndarray::view(1,5)(1,3,2)()]);          // (range, slice, full)
    CHECK_VIEW_RMC(array[ndarray::view(1,5)(1,3,2)(1)]);         // (range, slice, scalar)

    CHECK_VIEW_RMC(array[ndarray::view(1,5)(1,3)(0,2,2)]);       // (range, range, slice)
    CHECK_VIEW_RMC(array[ndarray::view(1,5)(1,3)(0,2)]);         // (range, range, range)
    CHECK_VIEW_RMC(array[ndarray::view(1,5)(1,3)()]);            // (range, range, full)
    CHECK_VIEW_RMC(array[ndarray::view(1,5)(1,3)(1)]);           // (range, range, scalar)

    CHECK_VIEW_RMC(array[ndarray::view(1,5)()(0,2,2)]);          // (range, range, slice)
    CHECK_VIEW_RMC(array[ndarray::view(1,5)()(0,2)]);            // (range, range, range)
    CHECK_VIEW_RMC(array[ndarray::view(1,5)()()]);               // (range, range, full)
    CHECK_VIEW_RMC(array[ndarray::view(1,5)()(1)]);              // (range, range, scalar)

    CHECK_VIEW_RMC(array[ndarray::view(1,5)(2)(0,2,2)]);         // (range, scalar, slice)
    CHECK_VIEW_RMC(array[ndarray::view(1,5)(2)(0,2)]);           // (range, scalar, range)
    CHECK_VIEW_RMC(array[ndarray::view(1,5)(2)()]);              // (range, scalar, full)

    CHECK_VIEW_RMC(array[ndarray::view()(1,3,2)(0,2,2)]);        // (full, slice, slice)
    CHECK_VIEW_RMC(array[ndarray::view()(1,3,2)(0,2)]);          // (full, slice, range)
    CHECK_VIEW_RMC(array[ndarray::view()(1,3,2)()]);             // (full, slice, full)
    CHECK_VIEW_RMC(array[ndarray::view()(1,3,2)(1)]);            // (full, slice, scalar)

    CHECK_VIEW_RMC(array[ndarray::view()(1,3)(0,2,2)]);          // (full, range, slice)
    CHECK_VIEW_RMC(array[ndarray::view()(1,3)(0,2)]);            // (full, range, range)
    CHECK_VIEW_RMC(array[ndarray::view()(1,3)()]);               // (full, range, full)
    CHECK_VIEW_RMC(array[ndarray::view()(1,3)(1)]);              // (full, range, scalar)

    CHECK_VIEW_RMC(array[ndarray::view()()(0,2,2)]);             // (full, range, slice)
    CHECK_VIEW_RMC(array[ndarray::view()()(0,2)]);               // (full, range, range)
    CHECK_VIEW_RMC(array[ndarray::view()()()]);                  // (full, range, full)
    CHECK_VIEW_RMC(array[ndarray::view()()(1)]);                 // (full, range, scalar)

    CHECK_VIEW_RMC(array[ndarray::view()(2)(0,2,2)]);            // (full, scalar, slice)
    CHECK_VIEW_RMC(array[ndarray::view()(2)(0,2)]);              // (full, scalar, range)
    CHECK_VIEW_RMC(array[ndarray::view()(2)()]);                 // (full, scalar, full)

    CHECK_VIEW_RMC(array[ndarray::view(3)(1,3,2)(0,2,2)]);        // (scalar, slice, slice)
    CHECK_VIEW_RMC(array[ndarray::view(3)(1,3,2)(0,2)]);          // (scalar, slice, range)
    CHECK_VIEW_RMC(array[ndarray::view(3)(1,3,2)()]);             // (scalar, slice, full)
    CHECK_VIEW_RMC(array[ndarray::view(3)(1,3,2)(1)]);            // (scalar, slice, scalar)

    CHECK_VIEW_RMC(array[ndarray::view(3)(1,3)(0,2,2)]);          // (scalar, range, slice)
    CHECK_VIEW_RMC(array[ndarray::view(3)(1,3)(0,2)]);            // (scalar, range, range)
    CHECK_VIEW_RMC(array[ndarray::view(3)(1,3)()]);               // (scalar, range, full)
    CHECK_VIEW_RMC(array[ndarray::view(3)(1,3)(1)]);              // (scalar, range, scalar)

    CHECK_VIEW_RMC(array[ndarray::view(3)()(0,2,2)]);             // (scalar, range, slice)
    CHECK_VIEW_RMC(array[ndarray::view(3)()(0,2)]);               // (scalar, range, range)
    CHECK_VIEW_RMC(array[ndarray::view(3)()()]);                  // (scalar, range, full)
    CHECK_VIEW_RMC(array[ndarray::view(3)()(1)]);                 // (scalar, range, scalar)

    CHECK_VIEW_RMC(array[ndarray::view(3)(2)(0,2,2)]);            // (scalar, scalar, slice)
    CHECK_VIEW_RMC(array[ndarray::view(3)(2)(0,2)]);              // (scalar, scalar, range)
    CHECK_VIEW_RMC(array[ndarray::view(3)(2)()]);                 // (scalar, scalar, full)

}
