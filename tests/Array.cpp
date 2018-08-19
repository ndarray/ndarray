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
#include "catch2/catch.hpp"
#include "ndarray/Array.hpp"

using namespace ndarray;

namespace {

template <typename T, Size N>
class TestIndexVector {
public:

    TestIndexVector(std::initializer_list<T> data) : _data(data) {}

    std::initializer_list<T> initializer_list() const { return _data; }

    std::vector<T> vector() const { return std::vector<T>(_data.begin(), _data.end()); }

    std::array<T, N> array() const {
        std::array<T, N> r;
        std::copy(_data.begin(), _data.end(), r.begin());
        return r;
    }

    T const & operator[](Size n) const { return _data.begin()[n]; }

private:
    std::initializer_list<T> _data;
};

template <Size N>
class TestStructure {
public:

    TestStructure(std::initializer_list<Size> shape_, std::initializer_list<Offset> strides_) :
        shape(shape_), strides(strides_), full_size(1)
    {
        for (Size i = 0; i < N; ++i) {
            full_size *= shape[i];
        }
    }

    template <typename T, Offset C>
    void check(Array<T, N, C> const & array) const {
        REQUIRE(array.shape() == shape.array());
        REQUIRE(array.strides() == strides.array());
        REQUIRE(array.full_size() == full_size);
        REQUIRE(array.size() == shape[0]);
        REQUIRE(array.stride() == strides[0]);
        REQUIRE(!array.empty());
    }

    template <typename T, Offset C>
    void runConstructionTest(MemoryOrder order) const {
        SECTION("Automatic strides with allocation") {
            check(Array<T, N, C>(shape.initializer_list(), order));
            check(Array<T, N, C>(shape.vector(), order));
            check(Array<T, N, C>(shape.array(), order));
        }
        using U = typename std::remove_const<T>::type;
        std::shared_ptr<T> data(new U[full_size], std::default_delete<U[]>());
        SECTION("Automatic strides without allocation") {
            check(Array<T, N, C>(data, shape.initializer_list(), order));
            check(Array<T, N, C>(data, shape.vector(), order));
            check(Array<T, N, C>(data, shape.array(), order));
        }
        SECTION("Explicit strides") {
            check(Array<T, N, C>(data, shape.initializer_list(), strides.initializer_list()));
            check(Array<T, N, C>(data, shape.vector(), strides.vector()));
            check(Array<T, N, C>(data, shape.vector(), strides.array()));
            check(Array<T, N, C>(data, shape.array(), strides.vector()));
            check(Array<T, N, C>(data, shape.array(), strides.array()));
        }
    }

    TestIndexVector<Size, N> shape;
    TestIndexVector<Offset, N> strides;
    Size full_size;
};

} // <anonymous>


TEST_CASE("Array: construction", "[Array]") {
    SECTION("Row-major contiguous") {
        TestStructure<3> s = {{4, 3, 2}, {24, 8, 4}};
        s.runConstructionTest<float, 3>(MemoryOrder::ROW_MAJOR);
        s.runConstructionTest<float const, 3>(MemoryOrder::ROW_MAJOR);
    }
    SECTION("Column-major contiguous") {
        TestStructure<3> s = {{4, 3, 2}, {8, 32, 96}};
        s.runConstructionTest<double, -3>(MemoryOrder::COL_MAJOR);
        s.runConstructionTest<double const, -3>(MemoryOrder::COL_MAJOR);
    }
}