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
#include "ndarray/detail/ArrayImpl.hpp"

using namespace ndarray;

namespace {

struct CountedElement {

    static int next;

    static void reset() { next = 0; }

    CountedElement() : value(next++) {}

    int value;
};

int CountedElement::next = 0;

template <typename T>
void check_index(detail::ArrayImpl<2> const & a, std::initializer_list<std::initializer_list<T>> const & b) {
    auto shape = a.layout->shape();
    for (Size i = 0; i < shape[0]; ++i) {
        for (Size j = 0; j < shape[1]; ++j) {
            std::array<Size, 2> indices = {i, j};
            T * v = reinterpret_cast<T*>(a.index(indices));
            REQUIRE(*v == b.begin()[i].begin()[j]);
        }
    }
}

template <Size N>
void check_comparisons(detail::ArrayImpl<N> const & a, detail::ArrayImpl<N> const & b) {
    REQUIRE(a == a);
    REQUIRE(b == b);
    REQUIRE(a == detail::ArrayImpl<N>(a));
    REQUIRE(b == detail::ArrayImpl<N>(b));
    REQUIRE(a == detail::ArrayImpl<N>(a.buffer, a.layout));
    REQUIRE(b == detail::ArrayImpl<N>(b.buffer, b.layout));
    REQUIRE(a != detail::ArrayImpl<N>());
    REQUIRE(b != detail::ArrayImpl<N>());
    REQUIRE(a != detail::ArrayImpl<N>(a.buffer, b.layout));
    REQUIRE(a != detail::ArrayImpl<N>(b.buffer, a.layout));
    REQUIRE(b != detail::ArrayImpl<N>(a.buffer, b.layout));
    REQUIRE(b != detail::ArrayImpl<N>(b.buffer, a.layout));
}

void check_index_pair(detail::ArrayImpl<2> const & rm, detail::ArrayImpl<2> const & cm) {
    check_index(rm, {{0, 1, 2},
                     {3, 4, 5}});
    check_index(cm, {{0, 2, 4},
                     {1, 3, 5}});
}

} // <anonymous>

TEST_CASE("detail::ArrayImpl: construct and allocate", "[detail][ArrayImpl]") {
    std::array<Size, 2> shape = {2, 3};
    CountedElement::reset();
    detail::ArrayImpl<2> rm(shape, MemoryOrder::ROW_MAJOR, detail::TypeTag<CountedElement>());
    CountedElement::reset();
    detail::ArrayImpl<2> cm(shape, MemoryOrder::COL_MAJOR, detail::TypeTag<CountedElement>());
    check_index_pair(rm, cm);
    check_comparisons(rm, cm);  // requires buffers to differ, so we only run this test here
}

TEST_CASE("detail::ArrayImpl: external data, automatic strides", "[detail][ArrayImpl]") {
    std::array<Size, 2> shape = {2, 3};
    CountedElement::reset();
    std::shared_ptr<CountedElement> data(new CountedElement[6], std::default_delete<CountedElement[]>());
    detail::ArrayImpl<2> rm(data, shape, MemoryOrder::ROW_MAJOR);
    detail::ArrayImpl<2> cm(data, shape, MemoryOrder::COL_MAJOR);
    check_index_pair(rm, cm);
}

TEST_CASE("detail::ArrayImpl: external data, explicit strides", "[detail][ArrayImpl]") {
    std::array<Size, 2> shape = {2, 3};
    Offset s = sizeof(CountedElement);
    std::array<Offset, 2> rm_strides = {s*3, s};
    std::array<Offset, 2> cm_strides = {s, s*2};
    CountedElement::reset();
    std::shared_ptr<CountedElement> data(new CountedElement[6], std::default_delete<CountedElement[]>());
    detail::ArrayImpl<2> rm(data, shape, rm_strides);
    detail::ArrayImpl<2> cm(data, shape, cm_strides);
    check_index_pair(rm, cm);
}

TEST_CASE("detail::ArrayImpl: external buffer, Layout", "[detail][ArrayImpl]") {
    std::array<Size, 2> shape = {2, 3};
    auto rm_layout = detail::Layout<2>::make(shape, sizeof(CountedElement), MemoryOrder::ROW_MAJOR);
    auto cm_layout = detail::Layout<2>::make(shape, sizeof(CountedElement), MemoryOrder::COL_MAJOR);
    CountedElement::reset();
    std::shared_ptr<CountedElement> data(new CountedElement[6], std::default_delete<CountedElement[]>());
    std::shared_ptr<Byte> buffer(data, reinterpret_cast<Byte*>(data.get()));
    detail::ArrayImpl<2> rm(buffer, rm_layout);
    detail::ArrayImpl<2> cm(buffer, cm_layout);
    check_index_pair(rm, cm);
}

