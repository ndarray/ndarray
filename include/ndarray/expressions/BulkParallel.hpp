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
#ifndef NDARRAY_EXPRESSIONS_BulkParallel_hpp_INCLUDED
#define NDARRAY_EXPRESSIONS_BulkParallel_hpp_INCLUDED

#include <future>
#include <vector>

#include "ndarray/expressions/Executor.hpp"

namespace ndarray {
namespace expressions {

/**
 * An Executor that traverses segments of its dimension in different threads.
 *
 * BulkParallelExecutor divides the dimension it is responsible for into
 * roughly equal segments and uses std::async and std::future to traverse
 * those segments in parallel while still guaranteeing that ReductionFunctions
 * are invoked deterministically.  Specifically, each segment/thread receives
 * a copy of the ReductionFunction created by ReductionFunction::split(), and
 * these are joined back together in the same order they were created.
 *
 * As this form of parallelization may be somewhat heavyweight, BulkParallel
 * should generally be used with either large segments (i.e. shape much larger
 * than `num_threads`) or with a nested SequentialExecutor for inner
 * dimensions.
 *
 * @sa BulkParallel
 */
template <typename Next>
class BulkParallelExecutor : public Executor<Next::ndim + 1, BulkParallelExecutor<Next>> {
public:

    static_assert(IsExecutor<Next>::value);

    BulkParallelExecutor(Next && next, Size num_threads=0) :
        _num_threads(num_threads),
        _next(std::move(next)) {}

    template <typename Traversal, typename ReductionFunction>
    bool execute(Size const * shape, Traversal traversal, ReductionFunction & function) const {
        Size const size = *shape;
        ++shape;

        auto to_launch = [shape, this](
            Size start, Size stop,
            Traversal && local_traversal,
            ReductionFunction && local_function
        ) -> ReductionFunction {
            local_traversal += start;
            for (Size k = start; k < stop; ++k, ++local_traversal) {
                if (!_next.execute(shape, local_traversal.evaluate(), local_function)) {
                    break;
                }
            }
            return local_function;
        };

        std::vector<std::future<ReductionFunction>> futures;
        if (_num_threads == 0) {
            futures.reserve(size);
            for (Size n = 0; n < size; ++n) {
                futures.push_back(
                    std::async(to_launch, n, n + 1, Traversal(traversal), function.split())
                );
            }
        } else {
            futures.reserve(_num_threads);;
            for (Size n_launched = 0, start = 0; n_launched < _num_threads; ++n_launched) {
                Size stop = start + (size - start)/(_num_threads - n_launched);
                futures.push_back(
                    std::async(to_launch, start, stop, Traversal(traversal), function.split())
                );
                start = stop;
            }
        }

        for (auto & future : futures) {
            if (!function.join(future.get())) {
                return false;
            }
        }
        return true;
    }

private:
    Size _num_threads;
    Next _next;
};


/**
 * An ExecutorFactory for parallel evaluation, intended for use with outer
 * dimensions.
 *
 * @sa BulkParallelExecutor
 */
class BulkParallel {
public:

    /**
     * Construct a factory.
     *
     * @param[in]  num_threads  Number of threads to use, or zero to use one
     *                          thread for every element in the traversed
     *                          dimension.
     */
    explicit BulkParallel(Size num_threads) : _num_threads(num_threads) {}

    template <typename Next>
    auto makeExecutor(Next && next) const {
        return BulkParallelExecutor<Next>(std::move(next), _num_threads);
    }

private:
    Size _num_threads;
};


} // namespace expressions
} // namespace ndarray

#endif // !NDARRAY_EXPRESSIONS_BulkParallel_hpp_INCLUDED
