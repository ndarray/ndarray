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
#ifndef NDARRAY_EXPRESSIONS_Executor_hpp_INCLUDED
#define NDARRAY_EXPRESSIONS_Executor_hpp_INCLUDED

#include <tuple>

#include "ndarray/common.hpp"
#include "ndarray/expressions/Expression.hpp"

namespace ndarray {
namespace expressions {


#ifdef DOXYGEN

/**
 * A specialized functor that summarizes expressions.
 *
 * ReductionFunction is an informal concept, not a true class; it exists only
 * in documentation, as a way to specify the operations that should be
 * supported by any type passed as the last argument to Executor::reduce().
 */
class ReductionFunction {
public:

    /**
     * Add the given scalar value from an Expression to the reduction.
     *
     * @return  Return true if the expression should continue to be evaluated,
     *          false if it can be short-circuited.
     */
    bool accumulate(Scalar x);

    /**
     * Return a new RedunctionFunction with a copy of this' state.
     *
     * If evaluation is not short-circuited, the returned ReductionFunction
     * will later be combined with this via a call to join().
     *
     * It must be safe to call accumulate() on this and the returned
     * ReductionFunction from different threads.
     *
     * @return A ReductionFunction of the same type as this.
     */
    ReductionFunction split() const;

    /**
     * Merge in results from a ReductionFunction created by split().
     *
     * @return  Return true if the expression should continue to be evaluated,
     *          false if it can be short-circuited.
     */
    bool join(ReductionFunction && other);

    /**
     * Extract final results from the ReductionFunction after expression
     * evaluation is complete.
     */
    Scalar finish() &&;

};


/**
 * A factory for Executor objects of a particular type.
 *
 * ExecutorFactory is an informal concept, not a true class; it exists only
 * in documentation.
 */
class ExecutorFactory {
public:

    /**
     * Construct an Executor instance.
     *
     * @param[in]  next  An existing Executor or ExecutorLeaf instance that
     *                   the returned Executor should call on every iteration.
     */
    template <typename Next>
    Executor makeExecutor(Next && next) const;

};

#endif // DOXYGEN


/**
 * A ReductionFunction that accumulates nothing and evaluates all expression
 * elements.
 */
class DummyReductionFunction {
public:

    template <typename T>
    bool accumulate(T x) { return true; }

    DummyReductionFunction split() const {}

    bool join(DummyReductionFunction && other) { return true; }

    void finish() &&;

};


/**
 * Sentinal class for nested expression execution.
 *
 * All Executors are templated on the Executor for the next dimension;
 * ExecutorLeaf is used to break this chain for the last dimension.
 */
class ExecutorLeaf {
public:

    /**
     * ExecutorLeaf is always applied after the last dimension; it does
     * not advance a Traversal.
     */
    static constexpr Size ndim = 0;

    /**
     * Apply the pseudo-executor to a scalar value.
     */
    template <typename Result, typename ReductionFunction>
    bool execute(Size const *, Result && result, ReductionFunction & function) const {
        return function.accumulate(std::forward<Result>(result));
    }

};


/**
 * CRTP base class for objects that evaluate Expressions.
 *
 * Executor template classes are always templated on the Executor for their
 * next dimension (which may be a completely different template class), until
 * the chain is broken by ExecutionLeaf at dimension 0.
 */
template <Size N, typename Derived>
class Executor {
public:

    /**
     * The Expression dimensionality of Expressions or Traversals that can be
     * evaluted by this object.
     */
    static constexpr Size ndim = N;

    /**
     * Summarize the given Traversal by applying a ReductionFunction to its
     * elements.
     *
     * @param[in]  shape  Pointer to the number of elements in the current
     *                    dimension, with size[1] the number of elements in
     *                    the next dimension.
     *
     * @param[in]  traversal  Traversal this Executor should recursively
     *                        evaluate.
     *
     * @param[in, out]  function  ReductionFunction to apply to each
     *                            expression element.
     *
     * @return  The result of calling `function.finish()` after processing all
     *          elements.
     */
    template <typename Traversal, typename ReductionFunction>
    auto reduce(Size const * shape, Traversal && traversal, ReductionFunction function) const {
        static_assert(Traversal::ndim == ndim,
                      "Traversal dimensionality does not match Evaluator dimensionality.");
        execute(shape, std::forward<Traversal>(traversal), function);
        return std::move(function).finish();
    }

    /**
     * Evaluate all elements of the given Traversal.
     *
     * @param[in]  shape  Pointer to the number of elements in the current
     *                    dimension, with size[1] the number of elements in
     *                    the next dimension.
     *
     * @param[in]  traversal  Traversal this Executor should recursively
     *                        evaluate.
     */
    template <typename Traversal>
    void run(Size const * shape, Traversal && traversal) const {
        reduce(shape, std::forward<Traversal>(traversal), DummyReductionFunction());
    }

    /**
     * Summarize the given Expression by applying a ReductionFunction to its
     * elements.
     *
     * @param[in]  expression  Expression this Executor should recursively
     *                         evaluate.
     *
     * @param[in, out]  function  ReductionFunction to apply to each
     *                            expression element.
     *
     * @return  The result of calling `function.finish()` after processing all
     *          elements.
     */
    template <typename ExprDerived, typename ReductionFunction>
    auto reduce(Expression<ndim, ExprDerived> const & expression, ReductionFunction function) const {
        std::array<Size, ndim> const shape = expression.full_shape();
        return reduce(shape.data(), expression.traverse(), function);
    }

    /**
     * Evaluate all elements of the given Expression.
     *
     * @param[in]  expression  Expression this Executor should recursively
     *                         evaluate.
     */
    template <typename ExprDerived>
    void run(Expression<ndim, ExprDerived> const & expression) const {
        reduce(expression, DummyReductionFunction());
    }

protected:

    /**
     * Recursively evaluate an Expression.
     *
     * This method must be implemented by all subclasses.  While it is
     * conceptually protected, the use of CRTP requires that subclasses either
     * make it public or make Expression a friend.
     *
     * @param[in]  shape  Pointer to the number of elements in the current
     *                    dimension, with size[1] the number of elements in
     *                    the next dimension.
     *
     * @param[in]  traversal  Traversal this Executor should evaluate (passing
     *                        the result to its nested Executor), and advance
     *                        `*shape` times.
     *
     * @param[in, out]  function  ReductionFunction to apply to each expression
     *                            element.
     *
     * @return  Return true if the expression should continue to be evaluated,
     *          false if it can be short-circuited.
     */
    template <typename Traversal, typename ReductionFunction>
    bool execute(Size const * shape, Traversal && traversal, ReductionFunction & function) const {
        return static_cast<Derived const &>(*this).execute(
            shape,
            std::forward<Traversal>(traversal),
            function
        );
    }

};

/**
 * Type traits struct that checks whether a type is an Executor or
 * ExecutorLeaf.
 */
template <typename T>
using IsExecutor = std::integral_constant<
    bool,
    std::is_base_of<Executor<T::ndim, T>, T>::value || std::is_same<T, ExecutorLeaf>::value
>;


/**
 * An ExecutorFactory that constructs multi-dimensional Executors of with
 * potentially different types for different dimensions.
 *
 * The makeExecutor() free function (which delegates to this class) should
 * generally be called instead of using MultiExecutorFactory director.
 */
template <typename ...Sequence>
class MultiExecutorFactory {
    using Tuple = std::tuple<Sequence...>;
public:

    /**
     * Dimensionality of Executors produced by this factory.
     */
    static constexpr Size ndim = sizeof...(Sequence);

    /**
     * Construct from a sequence of single-dimension Executor factories.
     */
    explicit MultiExecutorFactory(Sequence const & ...args) : _tuple(args...) {}

    /**
     * Construct an Executor instance.
     *
     * @param[in]  next  An existing Executor or ExecutorLeaf instance that
     *                   the returned Executor should call on every iteration.
     */
    template <typename Next>
    auto makeExecutor(Next && next) const {
        return _makeExecutor(DimensionIndex<0>{}, std::move(next));
    }

private:

    template <typename Next>
    auto _makeExecutor(DimensionIndex<ndim>, Next && next) const {
        return next;
    }

    template <Size J, typename Next>
    auto _makeExecutor(DimensionIndex<J>, Next && next) const {
        return std::get<J>(_tuple).makeExecutor(
            _makeExecutor(DimensionIndex<J + 1>{}, std::move(next))
        );
    }

    Tuple _tuple;

};


//@{
/**
 * Combine a sequence of ExecutorFactory objects into a single one.
 *
 * @param[in]  factories  Sequence of ExecutorFactory objects, ordered
 *                        from outer dimensions to inner dimensions.
 */
template <typename FactorySequence>
FactorySequence const & makeExecutorFactory(FactorySequence const & factories) {
    return factories;
}
template <typename ...FactorySequence>
auto makeExecutorFactory(FactorySequence const & ...factories) {
    return MultiExecutorFactory<FactorySequence...>(factories...);
}
//@}


/**
 * Create an Executor from a sequence of ExecutorFactories.
 *
 * @param[in]  factories  Sequence of ExecutorFactory objects, ordered
 *                        from outer dimensions to inner dimensions.
 */
template <typename ...FactorySequence>
auto makeExecutor(FactorySequence const & ...factories) {
    return makeExecutorFactory(factories...).makeExecutor(ExecutorLeaf());
}


} // namespace expressions
} // namespace ndarray

#endif // !NDARRAY_EXPRESSIONS_Executor_hpp_INCLUDED
