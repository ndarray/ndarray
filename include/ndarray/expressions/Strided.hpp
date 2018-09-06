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
class StridedExpression : public Expression<N, StridedExpression<T, N>> {
public:

    template <Size J, typename K=void> class Traversal;

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
        return StridedExpression(_pointer, shape, new_strides);
    }

    decltype(auto) traverse() const {
        return Traversal<0>(_pointer, _strides.data());
    }

private:
    T * _pointer;
    std::array<Size, N> _shape;
    std::array<Offset, N> _strides;
};


template <typename T, Size N>
template <typename K>
class StridedExpression<T, N>::Traversal<N - 1, K> {
public:

    static constexpr bool is_leaf = true;

    using Result = T &;

    Traversal(T * pointer, Offset const * stride) : _pointer(pointer), _stride(stride) {}

    void increment() { _pointer += *_stride; }

    Result evaluate() const { return *_pointer; }

private:
    T * _pointer;
    Offset const * _stride;
};


template <typename T, Size N>
template <Size J, typename K>
class StridedExpression<T, N>::Traversal {
public:

    static constexpr bool is_leaf = false;

    using Result = Traversal<J + 1>;

    Traversal(T * pointer, Offset const * stride) : _pointer(pointer), _stride(stride) {}

    void increment() { _pointer += *_stride; }

    Result evaluate() const { return Result(_pointer, _stride + 1); }

private:
    T * _pointer;
    Offset const * _stride;
};



} // namespace expressions
} // namespace ndarray

#endif // !NDARRAY_EXPRESSIONS_Strided_hpp_INCLUDED
