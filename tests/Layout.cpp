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
#include "ndarray/detail/Layout.hpp"

using namespace ndarray;

namespace {

template <Size N>
void check_get_dim(
    detail::Layout<N> const & layout,
    std::array<Size, N> const & shape,
    std::array<Offset, N> const & strides,
    std::integral_constant<Size, N>
) {
    // empty specialization to break check_get_dim recursion (below)
}

template <Size P, Size N>
void check_get_dim(
    detail::Layout<N> const & layout,
    std::array<Size, N> const & shape,
    std::array<Offset, N> const & strides,
    std::integral_constant<Size, P>
) {
    REQUIRE(detail::get_dim<P>(layout).size() == shape[P]);
    REQUIRE(detail::get_dim<P>(layout).stride() == strides[P]);
    check_get_dim(layout, shape, strides, std::integral_constant<Size, P+1>());
}

template <Size N>
void check_layout(
    detail::Layout<N> const & layout,
    std::array<Size, N> const & shape,
    std::array<Offset, N> const & strides
) {
    REQUIRE(layout == *detail::Layout<N>::make(shape, strides));
    REQUIRE(layout.shape() == shape);
    REQUIRE(layout.strides() == strides);
    REQUIRE(layout.full_size() == std::accumulate(shape.begin(), shape.end(),
                                                   static_cast<Size>(1), std::multiplies<Size>()));
    Size i = 0;
    layout.for_each_dim(
        [&i, &shape, &strides](auto const & d) {
            REQUIRE(d.size() == shape[i]);
            REQUIRE(d.stride() == strides[i] );
            ++i;
            return true;
        }
    );
    Size j = N - 1;
    layout.for_each_dim_r(
        [&j, &shape, &strides](auto const & d) {
            REQUIRE(d.size() == shape[j]);
            REQUIRE(d.stride() == strides[j]);
            --j;
            return true;
        }
    );
    check_get_dim(layout, shape, strides, std::integral_constant<Size, 0>());
}

} // <anonymous>

TEST_CASE("detail::Layout: automatic row-major contiguous strides", "[detail][Layout]") {
    Size const element_size = 2;
    std::array<Size, 3> shape = {3, 4, 5};
    std::array<Offset, 3> strides = {40, 10, 2};
    auto layout = detail::Layout<3>::make(shape, element_size, MemoryOrder::ROW_MAJOR);
    check_layout(*layout, shape, strides);
    REQUIRE_NOTHROW(layout->check_contiguousness<3>(element_size));
    REQUIRE_NOTHROW(layout->check_contiguousness<2>(element_size));
    REQUIRE_NOTHROW(layout->check_contiguousness<1>(element_size));
    REQUIRE_NOTHROW(layout->check_contiguousness<0>(element_size));
    REQUIRE_THROWS_AS(layout->check_contiguousness<-1>(element_size), NoncontiguousError);
    REQUIRE_THROWS_AS(layout->check_contiguousness<-2>(element_size), NoncontiguousError);
    REQUIRE_THROWS_AS(layout->check_contiguousness<-3>(element_size), NoncontiguousError);
}

TEST_CASE("detail::Layout: explicit partially row-major strides", "[detail][Layout]") {
    Size const element_size = 2;
    std::array<Size, 3> shape = {3, 4, 5};
    std::array<Offset, 3> strides = {80, 10, 2};
    auto layout = detail::Layout<3>::make(shape, strides);
    check_layout(*layout, shape, strides);
    REQUIRE_THROWS_AS(layout->check_contiguousness<3>(element_size), NoncontiguousError);
    REQUIRE_NOTHROW(layout->check_contiguousness<2>(element_size));
    REQUIRE_NOTHROW(layout->check_contiguousness<1>(element_size));
    REQUIRE_NOTHROW(layout->check_contiguousness<0>(element_size));
    REQUIRE_THROWS_AS(layout->check_contiguousness<-1>(element_size), NoncontiguousError);
    REQUIRE_THROWS_AS(layout->check_contiguousness<-2>(element_size), NoncontiguousError);
    REQUIRE_THROWS_AS(layout->check_contiguousness<-3>(element_size), NoncontiguousError);
}

TEST_CASE("detail::Layout: automatic column-major contiguous strides", "[detail][Layout]") {
    Size const element_size = 2;
    std::array<Size, 3> shape = {3, 4, 5};
    std::array<Offset, 3> strides = {2, 6, 24};
    auto layout = detail::Layout<3>::make(shape, element_size, MemoryOrder::COL_MAJOR);
    check_layout(*layout, shape, strides);
    REQUIRE_THROWS_AS(layout->check_contiguousness<3>(element_size), NoncontiguousError);
    REQUIRE_THROWS_AS(layout->check_contiguousness<2>(element_size), NoncontiguousError);
    REQUIRE_THROWS_AS(layout->check_contiguousness<1>(element_size), NoncontiguousError);
    REQUIRE_NOTHROW(layout->check_contiguousness<0>(element_size));
    REQUIRE_NOTHROW(layout->check_contiguousness<-1>(element_size));
    REQUIRE_NOTHROW(layout->check_contiguousness<-2>(element_size));
    REQUIRE_NOTHROW(layout->check_contiguousness<-3>(element_size));
}

TEST_CASE("detail::Layout: explicit partially column-major strides", "[detail][Layout]") {
    Size const element_size = 2;
    std::array<Size, 3> shape = {3, 4, 5};
    std::array<Offset, 3> strides = {2, 6, 48};
    auto layout = detail::Layout<3>::make(shape, strides);
    check_layout(*layout, shape, strides);
    REQUIRE_THROWS_AS(layout->check_contiguousness<3>(element_size), NoncontiguousError);
    REQUIRE_THROWS_AS(layout->check_contiguousness<2>(element_size), NoncontiguousError);
    REQUIRE_THROWS_AS(layout->check_contiguousness<1>(element_size), NoncontiguousError);
    REQUIRE_NOTHROW(layout->check_contiguousness<0>(element_size));
    REQUIRE_NOTHROW(layout->check_contiguousness<-1>(element_size));
    REQUIRE_NOTHROW(layout->check_contiguousness<-2>(element_size));
    REQUIRE_THROWS_AS(layout->check_contiguousness<-3>(element_size), NoncontiguousError);
}

TEST_CASE("detail::Layout: explicit noncontiguous strides", "[detail][Layout]") {
    Size const element_size = 2;
    std::array<Size, 3> shape = {3, 4, 5};
    std::array<Offset, 3> strides = {4, 60, 12};
    auto layout = detail::Layout<3>::make(shape, strides);
    check_layout(*layout, shape, strides);
    REQUIRE_THROWS_AS(layout->check_contiguousness<3>(element_size), NoncontiguousError);
    REQUIRE_THROWS_AS(layout->check_contiguousness<2>(element_size), NoncontiguousError);
    REQUIRE_THROWS_AS(layout->check_contiguousness<1>(element_size), NoncontiguousError);
    REQUIRE_NOTHROW(layout->check_contiguousness<0>(element_size));
    REQUIRE_THROWS_AS(layout->check_contiguousness<-1>(element_size), NoncontiguousError);
    REQUIRE_THROWS_AS(layout->check_contiguousness<-2>(element_size), NoncontiguousError);
    REQUIRE_THROWS_AS(layout->check_contiguousness<-3>(element_size), NoncontiguousError);
}
