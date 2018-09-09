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
#ifndef NDARRAY_EXPRESSIONS_Sequential_hpp_INCLUDED
#define NDARRAY_EXPRESSIONS_Sequential_hpp_INCLUDED

#include "ndarray/expressions/Executor.hpp"

namespace ndarray {
namespace expressions {


/**
 * An Executor that simply iterates over its dimension sequentially with no
 * parallelism or vectorization.
 *
 * @sa Sequential
 */
template <typename Next>
class SequentialExecutor : public Executor<Next::ndim + 1, SequentialExecutor<Next>> {
public:

    static_assert(IsExecutor<Next>::value);

    explicit SequentialExecutor(Next && next) : _next(std::move(next)) {}

    template <typename Traversal, typename ReductionFunction>
    bool execute(Size const * shape, Traversal traversal, ReductionFunction & function) const {
        Size const size = *shape;
        for (Size k = 0u; k < size; ++k, ++traversal) {
            if (!_next.execute(shape + 1, traversal.evaluate(), function)) {
                return false;
            }
        }
        return true;
    }

private:
    Next _next;
};


/**
 * An ExecutorFactory for simple, sequential iteration.
 *
 * @sa SequentialExecutor
 */
class Sequential {
public:

    /**
     * Construct a factory.
     */
    Sequential() = default;

    template <typename Next>
    auto makeExecutor(Next && next) const {
        return SequentialExecutor<Next>(std::move(next));
    }

};


} // namespace expressions
} // namespace ndarray

#endif // !NDARRAY_EXPRESSIONS_Sequential_hpp_INCLUDED
