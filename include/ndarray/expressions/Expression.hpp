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
#ifndef NDARRAY_EXPRESSIONS_Expression_hpp_INCLUDED
#define NDARRAY_EXPRESSIONS_Expression_hpp_INCLUDED

#include <array>
#include <type_traits>

#include "ndarray/common.hpp"
#include "ndarray/errors.hpp"

namespace ndarray {
namespace expressions {

/**
 * Integer tag class for recursive iteration over expression dimensions.
 *
 * Using a subclass of std::integral_constant provides a little type safety
 * when multiple constexpr integers are in play in a single function.
 */
template <Size M>
struct DimensionIndex : public std::integral_constant<Size, M> {};


/**
 * CRTP base class for expression templates.
 *
 * Unlike Arrays, Expressions are not expected to own their data; they are
 * expected to be created, used, and destroyed with in a single statement.
 * They should be be copyable and moveable, but with a pointer/iterator-like
 * approach to constness and ownership; expressions are lightweight, and
 * copies should copy pointers and references rather than their targets.
 */
template <Size N, typename Derived>
class Expression {
public:

    /**
     * The number of dimensions of the expression.
     */
    static constexpr Size ndim = N;

    /**
     * Return the size of the jth dimension.
     *
     * Must be implemented by all subclasses.
     */
    template <Size J>
    Size shape_at(DimensionIndex<J> j) const {
        return static_cast<Derived const &>(*this).shape_at(j);
    }

    /**
     * Return a version of the expression with the given shape.
     *
     * When the current expression has a dimension with a size of one, and the
     * given shape has a larger size for that dimension, the returned
     * expression should be able to use the larger size by repeating the
     * single value.  When the expression cannot be mapped to the given shape,
     * subclasses should invoke Error::INCOMPATIBLE_ARGUMENTS.  When the
     * expression cannot be mapped to the given number of dimensions,
     * subclasses should static_assert.
     *
     * Must be implemented by all subclasses.
     *
     * @param[in]  shape   Shape the returned expression must have.
     *
     * @returns A new Expression instance with unspecified exact type.
     */
    template <Size M>
    decltype(auto) broadcast(std::array<Size, M> const & shape) && {
        return static_cast<Derived &&>(std::move(*this)).broadcast(shape);
    }

    /**
     * Return a Traversal for the first dimension of the expression.
     *
     * Must be implemented by all subclasses.
     *
     * @returns A new Traversal instance with unspecified exact type.
     */
    decltype(auto) traverse() const {
        return static_cast<Derived const &>(*this).traverse();
    }

    /**
     * Return the full shape of the expression.
     */
    std::array<Size, ndim> full_shape() const {
        std::array<Size, ndim> result = {0};
        _set_shape(result, DimensionIndex<0>{});
        return result;
    }

private:

    void _set_shape(std::array<Size, ndim> & out, DimensionIndex<ndim>) const {}

    template <Size J>
    void _set_shape(std::array<Size, ndim> & out, DimensionIndex<J> j) const {
        out[J] = shape_at(j);
        NDARRAY_ASSERT_CHECK(out[J] != 0, Error::INCOMPATIBLE_ARGUMENTS,
                             "Expression does not define a size for index {:d}.", J);
        _set_shape(out, DimensionIndex<J + 1>{});
    }

};


#ifdef DOXYGEN

/**
 * A nested iterator-like quantity used to traverse expressions.
 *
 * Traversal is an informal concept, not a true class; it exists only in
 * documentation, as a way to specify the operations that should be supported
 * by any type returned by Expression::traverse().
 */
class Traversal {
public:

    /**
     * The type returned by evaluate().
     *
     * When ndim > 1, this should be another Traversal type.  When ndim == 1,
     * this should be a scalar type.
     */
    using Result = /* unspecified */;

    /**
     * The dimension of the expression this traversal iterates over.
     */
    static constexpr ndim;

    /**
     * Advance the traversal by one element in its dimension.
     */
    Traversal & operator++();

    /**
     * Advance the traversal by n elements in its dimension.
     */
    Traversal & operator+=(Offset n);

    /**
     * Return a Traversal to the next dimension (ndim > 1) or a scalar value
     * or reference (ndim == 1).
     */
    Result evaluate();

};

#endif // DOXYGEN

/**
 * Type traits struct that yields the outermost Traversal type for an
 * Expression.
 */
template <typename Expression>
struct GetTraversal {
    using Type = decltype(std::declval<Expression>().traversal());
};

/**
 * Helper declaration for GetTraversalType.
 */
template <typename Expression>
using GetTraversalType = typename GetTraversal<Expression>::Type;


} // namespace expressions
} // namespace ndarray

#endif // !NDARRAY_EXPRESSIONS_Expression_hpp_INCLUDED
