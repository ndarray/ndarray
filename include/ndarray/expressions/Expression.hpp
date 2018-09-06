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


template <Size M>
struct DimensionIndex : public std::integral_constant<Size, M> {};


template <Size N, typename Derived>
class Expression {
public:

    static constexpr Size ndim = N;

    template <Size J>
    Size shape_at(DimensionIndex<J> j) const {
        return static_cast<Derived const &>(*this).shape_at(j);
    }

    template <Size M>
    decltype(auto) broadcast(std::array<Size, M> const & shape) && {
        return static_cast<Derived &&>(std::move(*this)).broadcast(shape);
    }

    decltype(auto) traverse() const {
        return static_cast<Derived const *>(*this).traverse();
    }

    std::array<Size, ndim> full_shape() const {
        std::array<Size, ndim> const result = {0};
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
    }

};


#ifdef NDARRAY_DOCUMENTATION_ONLY

class Traversal {
public:

    static constexpr bool is_leaf = /* unspecified */;

    using Result = /* unspecified */;

    void increment();

    Result evaluate();

};

#endif // NDARRAY_DOCUMENTATION_ONLY


template <typename Expression>
struct GetTraversal {
    using Type = decltype(std::declval<Expression>().traversal());
};

template <typename Expression>
using GetTraversalType = typename GetTraversal<Expression>::Type;


} // namespace expressions
} // namespace ndarray

#endif // !NDARRAY_EXPRESSIONS_Expression_hpp_INCLUDED
