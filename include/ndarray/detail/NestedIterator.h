// -*- c++ -*-
/*
 * Copyright (c) 2010-2012, Jim Bosch
 * All rights reserved.
 *
 * ndarray is distributed under a simple BSD-like license;
 * see the LICENSE file that should be present in the root
 * of the source distribution, or alternately available at:
 * https://github.com/ndarray/ndarray
 */
#ifndef NDARRAY_DETAIL_NestedIterator_h_INCLUDED
#define NDARRAY_DETAIL_NestedIterator_h_INCLUDED

/** 
 *  @file ndarray/detail/NestedIterator.h
 *
 *  @brief Definition of NestedIterator.
 */

#include "ndarray_fwd.h"

namespace ndarray {
namespace detail {

/**
 *  @internal @brief Nested-array iterator class for Array with ND > 1.
 *
 *  NestedIterator does not support multi-pass algorithms, so it is formally
 *  just an InputIterator.  It nevertheless supports efficient indexing,
 *  arithmetic, and comparison operations of the sort provided by
 *  RandomAccessIterator.
 *
 *  @ingroup ndarrayInternalGroup
 */
template <typename T, int N, int C>
class NestedIterator {
public:

    typedef typename ArrayTraits<T,N,C>::Value value_type;
    typedef typename ArrayTraits<T,N,C>::Reference const & reference;
    typedef typename ArrayTraits<T,N,C>::Reference const * pointer;
    typedef Offset difference_type;
    typedef std::input_iterator_tag iterator_category;

    typedef typename ArrayTraits<T,N,C>::Value Value;
    typedef typename ArrayTraits<T,N,C>::Reference Reference;

    NestedIterator() : _ref(Value()), _stride(0) {}

    NestedIterator(Reference const & ref, Offset stride) : _ref(ref), _stride(stride) {}

    NestedIterator(NestedIterator const & other) : _ref(other._ref), _stride(other._stride) {}

    template <typename T_, int C_>
    NestedIterator(NestedIterator<T_,N,C_> const & other) : _ref(other._ref), _stride(other._stride) {}

    // Return Reference (ArrayRef) instead of Value (Array) so
    // `iter[n] = <expr>` copies values, not pointers.  Can't
    // return (lowercase) reference because it's a temporary.
    Reference operator[](Size n) const {
        Reference r(_ref);
        r._data += n * _stride;
        return r;
    }

    reference operator*() const { return _ref; }

    pointer operator->() { return &_ref; }

    NestedIterator & operator=(NestedIterator const & other) {
        if (&other != this) {
            _ref._data = other._ref._data;
            _ref._core = other._ref._core;
            _stride = other._stride;
        }
        return *this;
    }

    template <typename T_, int C_>
    NestedIterator & operator=(NestedIterator<T_,N,C_> const & other) {
        _ref._data = other._ref._data;
        _ref._core = other._ref._core;
        _stride = other._stride;
        return *this;
    }

    template <typename T_, int C_>
    bool operator==(NestedIterator<T_,N,C_> const & other) const {
        return _ref._data == other._ref._data;
    }

    template <typename T_, int C_>
    bool operator!=(NestedIterator<T_,N,C_> const & other) const {
        return !(*this == other);
    }

    template <typename T_, int C_>
    bool operator<(NestedIterator<T_,N,C_> const & other) const {
        return _ref._data < other._ref._data;
    }

    template <typename T_, int C_>
    bool operator>(NestedIterator<T_,N,C_> const & other) const {
        return _ref._data > other._ref._data;
    }

    template <typename T_, int C_>
    bool operator<=(NestedIterator<T_,N,C_> const & other) const {
        return _ref._data <= other._ref._data;
    }

    template <typename T_, int C_>
    bool operator>=(NestedIterator<T_,N,C_> const & other) const {
        return _ref._data >= other._ref._data;
    }

    NestedIterator & operator++() {
        _ref._data += _stride;
        return *this;
    }

    NestedIterator & operator--() {
        _ref._data -= _stride;
        return *this;
    }

    NestedIterator operator++(int) {
        NestedIterator copy(*this);
        ++(*this);
        return copy;
    }

    NestedIterator operator--(int) {
        NestedIterator copy(*this);
        --(*this);
        return copy;
    }

    NestedIterator & operator+=(difference_type n) {
        _ref._data += _stride*n;
        return *this;
    }

    NestedIterator & operator-=(difference_type n) {
        _ref._data -= _stride*n;
        return *this;
    }

    NestedIterator operator+(difference_type n) {
        NestedIterator copy(*this);
        copy += n;
        return copy;
    }

    NestedIterator operator-(difference_type n) {
        NestedIterator copy(*this);
        copy -= n;
        return copy;
    }

    friend NestedIterator operator+(difference_type n, NestedIterator const & self) {
        NestedIterator copy(self);
        copy += n;
        return copy;
    }

    template <typename T_, int C_>
    difference_type operator-(NestedIterator<T_,N,C_> const & other) {
        return (_ref.data - other._ref._data) / _stride;
    }

private:

    template <typename T_, int N_, int C_> friend class NestedIterator;

    Reference _ref;
    Offset _stride;
};

} // namespace detail
} // namespace ndarray

#endif // !NDARRAY_DETAIL_NestedIterator_h_INCLUDED
