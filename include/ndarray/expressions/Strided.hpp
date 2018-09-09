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
#ifndef NDARRAY_EXPRESSIONS_Strided_hpp_INCLUDED
#define NDARRAY_EXPRESSIONS_Strided_hpp_INCLUDED

#include <array>
#include <type_traits>

#include "ndarray/expressions/Expression.hpp"

namespace ndarray {
namespace expressions {


template <typename T, Size N>
class StridedTraversal;


/**
 * Sentinal 1-d specialization of StridedTraversal.
 */
template <typename T>
class StridedTraversal<T, 1> {
public:

    static constexpr Size ndim = 1;

    using Result = T &;

    StridedTraversal(T * pointer, Offset const * strides) : _pointer(pointer), _strides(strides) {}

    StridedTraversal & operator++() {
        _pointer += *_strides;
        return *this;
    }

    StridedTraversal & operator+=(Offset n) {
        _pointer += (*_strides)*n;
        return *this;
    }

    Result evaluate() const { return *_pointer; }

private:
    T * _pointer;
    Offset const * _strides;
};


/**
 * A Traversal for strided arrays.
 */
template <typename T, Size N>
class StridedTraversal {
public:

    static constexpr Size ndim = N;

    using Result = StridedTraversal<T, N - 1>;

    StridedTraversal(T * pointer, Offset const * strides) : _pointer(pointer), _strides(strides) {}

    StridedTraversal & operator++() {
        _pointer += *_strides;
        return *this;
    }

    StridedTraversal & operator+=(Offset n) {
        _pointer += (*_strides)*n;
        return *this;
    }

    Result evaluate() const { return Result(_pointer, _strides + 1); }

private:
    T * _pointer;
    Offset const * _strides;
};


/**
 * An Expression for strided arrays.
 */
template <typename T, Size N>
class StridedExpression : public Expression<N, StridedExpression<T, N>> {
public:

    using Traversal = StridedTraversal<T, N>;

    StridedExpression(
        T * pointer,
        std::array<Size, N> const & shape,
        std::array<Offset, N> const & strides
    ) : _pointer(pointer),
        _shape(shape),
        _strides(strides)
    {}

    template <Size J>
    Size shape_at(DimensionIndex<J> j) const { return _shape[J]; }

    template <Size M>
    decltype(auto) broadcast(std::array<Size, M> const & shape) && {
        static_assert(M >= N, "Cannot decrease dimensionality in broadcast.");
        std::array<Size, M> new_strides = {0};
        for (Size j = 0u; j < N; ++j) {
            if (_shape[j] != shape[j]) {
                if (_shape[j] == 1u) {
                    new_strides[j] = 0;
                } else {
                    NDARRAY_FAIL(Error::INCOMPATIBLE_ARGUMENTS,
                                 "Shapes {:d} and {:d} in dimension {:d} cannot be broadcast together.",
                                 _shape[j], shape[j], j);
                }
            }
        }
        return StridedExpression<T, M>(_pointer, shape, new_strides);
    }

    decltype(auto) traverse() const {
        return Traversal(_pointer, _strides.data());
    }

private:
    T * _pointer;
    std::array<Size, N> _shape;
    std::array<Offset, N> _strides;
};



} // namespace expressions
} // namespace ndarray

#endif // !NDARRAY_EXPRESSIONS_Strided_hpp_INCLUDED
