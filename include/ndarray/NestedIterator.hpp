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
#ifndef NDARRAY_NestedIterator_hpp_INCLUDED
#define NDARRAY_NestedIterator_hpp_INCLUDED

#include <iterator>
#include <utility>

#include "ndarray/common.hpp"
#include "ndarray/errors.hpp"

namespace ndarray {

template <typename T, Size N, Offset C>
class NestedIterator {
public:

    using value_type = Array<T, N, C>;
    using reference = value_type const &;
    using pointer = value_type const *;
    using difference_type = Offset;
    using iterator_category = std::input_iterator_tag;  // would be random-access, but not multi-pass

    NestedIterator() : _array() {}

    explicit NestedIterator(Array<T, N, C> array) : _array(std::move(array)) {}

    NestedIterator(NestedIterator const &) noexcept = default;
    NestedIterator(NestedIterator &&) noexcept = default;

    NestedIterator & operator=(NestedIterator const &) noexcept = default;
    NestedIterator & operator=(NestedIterator &&) noexcept = default;

    ~NestedIterator() noexcept = default;

    void swap(NestedIterator & other) noexcept {
        using namespace std;
        swap(_array, other._array);
    }

    friend void swap(NestedIterator & a, NestedIterator & b) noexcept {
        a.swap(b);
    }

    template <typename T2, Offset C2>
    bool operator==(NestedIterator<T2, N, C2> const & rhs) const { return _array == rhs._array; }

    template <typename T2, Offset C2>
    bool operator!=(NestedIterator<T2, N, C2> const & rhs) const { return !(*this == rhs); }

    template <typename T2, Offset C2>
    bool operator<(NestedIterator<T2, N, C2> const & rhs) const {
        NDARRAY_ASSERT_CHECK(_array.data() != nullptr, Error::UNINITIALIZED, "null iterator comparison");
        NDARRAY_ASSERT_CHECK(rhs._array.data() != nullptr, Error::UNINITIALIZED, "null iterator comparison");
        return _array.data() < rhs._array.data();
    }

    template <typename T2, Offset C2>
    bool operator>(NestedIterator<T2, N, C2> const & rhs) const {
        NDARRAY_ASSERT_CHECK(_array.data() != nullptr, Error::UNINITIALIZED, "null iterator comparison");
        NDARRAY_ASSERT_CHECK(rhs._array.data() != nullptr, Error::UNINITIALIZED, "null iterator comparison");
        return _array.data() > rhs._array.data();
    }

    template <typename T2, Offset C2>
    bool operator<=(NestedIterator<T2, N, C2> const & rhs) const {
        NDARRAY_ASSERT_CHECK(_array.data() != nullptr, Error::UNINITIALIZED, "null iterator comparison");
        NDARRAY_ASSERT_CHECK(rhs._array.data() != nullptr, Error::UNINITIALIZED, "null iterator comparison");
        return _array.data() <= rhs._array.data();
    }

    template <typename T2, Offset C2>
    bool operator>=(NestedIterator<T2, N, C2> const & rhs) const {
        NDARRAY_ASSERT_CHECK(_array.data() != nullptr, Error::UNINITIALIZED, "null iterator comparison");
        NDARRAY_ASSERT_CHECK(rhs._array.data() != nullptr, Error::UNINITIALIZED, "null iterator comparison");
        return _array.data() >= rhs._array.data();
    }

    reference operator*() const { return _array; }

    pointer operator->() const { return &_array; }

    NestedIterator & operator++() { return _advance(1); }

    NestedIterator & operator--() { return _advance(-1); }

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

    NestedIterator & operator+=(difference_type n) { return _advance(n); }

    NestedIterator & operator-=(difference_type n) { return _advance(-n); }

    NestedIterator operator+(difference_type n) const {
        return NestedIterator(*this) += n;
    }

    NestedIterator operator-(difference_type n) const {
        return NestedIterator(*this) -= n;
    }

    friend NestedIterator operator+(difference_type n, NestedIterator iter) {
        return iter + n;
    }

    template <typename T2, Offset C2>
    difference_type operator-(NestedIterator<T2, N, C2> const & rhs) const {
        NDARRAY_ASSERT_CHECK(_array.data() != nullptr, Error::UNINITIALIZED, "null iterator comparison");
        NDARRAY_ASSERT_CHECK(rhs._array.data() != nullptr, Error::UNINITIALIZED, "null iterator comparison");
        NDARRAY_ASSERT_CHECK(_stride() == rhs._stride(), Error::INCOMPATIBLE_ARGUMENTS,
                             "iterators have different strides");
        NDARRAY_ASSERT_CHECK(_stride() != 0, Error::UNINITIALIZED, "iterator stride is zero");
        NDARRAY_ASSERT_CHECK((_array._impl.data() - rhs._array._impl.data()) % _stride() == 0,
                             Error::INCOMPATIBLE_ARGUMENTS,
                             "iterator pointer offset is not a multiple of stride");
        return (_array._impl.data() - rhs._array._impl.data())/_stride();
    }

    value_type operator[](difference_type n) const {
        return *NestedIterator(*this)._advance(n);
    }

private:

    template <typename T2, Size N2, Offset C2> friend class NestedIterator;

    Offset _stride() const {
        return static_cast<detail::Layout<N+1> const &>(*_array._impl.layout).stride();
    }

    NestedIterator & _advance(Offset n) {
        NDARRAY_ASSERT_CHECK(_array._impl.buffer != nullptr, Error::UNINITIALIZED,
                             "null iterator incremented");
        _array._impl.buffer = std::shared_ptr<Byte>(
            _array._impl.buffer,
            _array._impl.buffer.get() + n*_stride()
        );
        return *this;
    }

    Array<T, N, C> _array;
};

} // ndarray

#endif // !NDARRAY_NestedIterator_hpp_INCLUDED
