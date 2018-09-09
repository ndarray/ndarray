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
#include <vector>
#include <mutex>
#include <thread>
#include <chrono>
#include "catch2/catch.hpp"

#define NDARRAY_ASSERT_AUDIT_ENABLED true

#include "ndarray/expressions/Strided.hpp"
#include "ndarray/expressions/Sequential.hpp"
#include "ndarray/expressions/BulkParallel.hpp"

using namespace ndarray;
using namespace ndarray::expressions;


namespace {

/*
 * Data for the tests below: a strided, row-major contiguous 3-d array
 * with shape (2, 3, 4) that starts from zero and increments by one to 23.
 */
struct StridedTestData {

    StridedTestData() {
        std::iota(values.begin(), values.end(), 0);
    }

    auto expr() const {
        std::array<Offset, 3> strides = {12, 4, 1};
        return StridedExpression<int const, 3>(values.data(), shape, strides);
    }

    std::array<Size, 3> shape = {2, 3, 4};
    std::array<int, 2*3*4> values;
};

/**
 * A ReductionFunctor that records all values it sees in a std::vector,
 * allowing test code to verify what an expression generated.
 */
template <typename T>
struct Recorder {

    Recorder() = default;

    bool accumulate(T x) {
        _values.push_back(x);
        return true;
    }

    Recorder split() const {
        return Recorder(*this);
    }

    bool join(Recorder && other) {
        _values.insert(_values.end(), other._values.begin(), other._values.end());
        return true;
    }

    std::vector<T> finish() && {
        return std::move(_values);
    }

private:
    std::vector<T> _values;
};


/*
 * A version of Recorder designed to test asynchronous expression execution.
 * Instances operate on a shared std::vector, and values below the given
 * cutoff value are not appended until the vector's size is above the given
 * cutoff size.
 */
template <typename T>
struct WaitingRecorder {

    explicit WaitingRecorder(T cutoff_value, Size cutoff_size) :
        _cutoff_value(cutoff_value),
        _cutoff_size(cutoff_size),
        _data(std::make_shared<Data>())
    {}

    bool accumulate(T x) {
        using namespace std::chrono_literals;
        if (x < _cutoff_value) {
            while (true) {
                std::this_thread::sleep_for(5ms);
                std::lock_guard<std::mutex> guard(_data->mutex);
                if (_data->values.size() >= _cutoff_size) {
                    break;
                }
            }
        }
        std::lock_guard<std::mutex> guard(_data->mutex);
        _data->values.push_back(x);
        return true;
    }

    WaitingRecorder split() const {
        return WaitingRecorder(*this);
    }

    bool join(WaitingRecorder && other) {
        return true;
    }

    std::vector<T> finish() && {
        return std::move(_data->values);
    }

private:

    struct Data {
        std::mutex mutex;
        std::vector<T> values;
    };

    T _cutoff_value;
    Size _cutoff_size;
    std::shared_ptr<Data> _data;
};

} // anonymous


TEST_CASE("expressions: Strided", "[expressions][Strided]") {
    StridedTestData data;
    auto expr = data.expr();
    Size n = 0;
    auto t0 = expr.traverse();
    for (Size k0 = 0; k0 < data.shape[0]; ++k0) {
        auto t1 = t0.evaluate();
        for (Size k1 = 0; k1 < data.shape[1]; ++k1) {
            auto t2 = t1.evaluate();
            for (Size k2 = 0; k2 < data.shape[2]; ++k2) {
                REQUIRE(t2.evaluate() == n);
                ++n;
                ++t2;
            }
            ++t1;
        }
        ++t0;
    }
}


TEST_CASE("expressions: Sequential", "[expressions][Sequential]") {
    StridedTestData data;
    auto expr = data.expr();
    auto executor = makeExecutor(Sequential(), Sequential(), Sequential());
    static_assert(decltype(executor)::ndim == 3);
    auto values = executor.reduce(expr, Recorder<int>());
    REQUIRE(values.size() == data.values.size());
    REQUIRE(std::equal(values.begin(), values.end(), data.values.begin()));
}


#if NDARRAY_TEST_WITH_THREADS

TEST_CASE("expressions: BulkParallel", "[expressions][BulkParallel]") {
    StridedTestData data;
    auto expr = data.expr();
    auto executor = makeExecutor(BulkParallel(2), Sequential(), Sequential());
    static_assert(decltype(executor)::ndim == 3);
    // Recorder properly copies its state in split() and integrates it in
    // join(), which ensures that even parallel execution maintains the order
    // in which expression values are seen.
    auto values1 = executor.reduce(expr, Recorder<int>());
    REQUIRE(values1.size() == data.values.size());
    REQUIRE(std::equal(values1.begin(), values1.end(), data.values.begin()));
    // WaitingRecorder shares state in split() and waits in accumulate in a way
    // that should guarantee that we record the values 12-23 before 0-11.
    auto values2 = executor.reduce(expr, WaitingRecorder<int>(12, 12));
    REQUIRE(values2.size() == data.values.size());
    auto mid = values2.begin() + 12;
    REQUIRE(std::equal(values2.begin(), mid, data.values.begin() + 12));
    REQUIRE(std::equal(mid, values2.end(), data.values.begin()));
}

#endif // NDARRAY_TEST_WITH_THREADS
