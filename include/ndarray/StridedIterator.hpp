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
#ifndef NDARRAY_StridedIterator_hpp_INCLUDED
#define NDARRAY_StridedIterator_hpp_INCLUDED

#include <iterator>
#include <utility>

#include "ndarray/common.hpp"
#include "ndarray/errors.hpp"

namespace ndarray {

template <typename T>
class StridedIterator {
public:

    using value_type = T;
    using reference = T &;
    using pointer = T *;
    using difference_type = Offset;
    using iterator_category = std::random_access_iterator_tag;

    StridedIterator() : _ptr(nullptr), _stride(0) {}

    StridedIterator(Byte * ptr, Offset stride) : _ptr(ptr), _stride(stride) {}

    StridedIterator(StridedIterator const &) noexcept = default;
    StridedIterator(StridedIterator &&) noexcept = default;

    StridedIterator & operator=(StridedIterator const &) noexcept = default;
    StridedIterator & operator=(StridedIterator &&) noexcept = default;

    ~StridedIterator() noexcept = default;

    void swap(StridedIterator & other) noexcept {
        using namespace std;
        swap(_ptr, other._ptr);
        swap(_stride, other._stride);
    }

    friend void swap(StridedIterator & a, StridedIterator & b) noexcept {
        a.swap(b);
    }

    template <typename U>
    bool operator==(StridedIterator<U> const & rhs) const { return _ptr == rhs.ptr; }

    template <typename U>
    bool operator!=(StridedIterator<U> const & rhs) const { return !(*this == rhs); }

    template <typename U>
    bool operator<(StridedIterator<U> const & rhs) const {
        NDARRAY_ASSERT_CHECK(_ptr != nullptr, Error::UNINITIALIZED, "null iterator comparison");
        NDARRAY_ASSERT_CHECK(rhs._ptr != nullptr, Error::UNINITIALIZED, "null iterator comparison");
        return _ptr < rhs._ptr;
    }

    template <typename U>
    bool operator>(StridedIterator<U> const & rhs) const {
        NDARRAY_ASSERT_CHECK(_ptr != nullptr, Error::UNINITIALIZED, "null iterator comparison");
        NDARRAY_ASSERT_CHECK(rhs._ptr != nullptr, Error::UNINITIALIZED, "null iterator comparison");
        return _ptr > rhs._ptr;
    }

    template <typename U>
    bool operator<=(StridedIterator<U> const & rhs) const {
        NDARRAY_ASSERT_CHECK(_ptr != nullptr, Error::UNINITIALIZED, "null iterator comparison");
        NDARRAY_ASSERT_CHECK(rhs._ptr != nullptr, Error::UNINITIALIZED, "null iterator comparison");
        return _ptr <= rhs._ptr;
    }

    template <typename U>
    bool operator>=(StridedIterator<U> const & rhs) const {
        NDARRAY_ASSERT_CHECK(_ptr != nullptr, Error::UNINITIALIZED, "null iterator comparison");
        NDARRAY_ASSERT_CHECK(rhs._ptr != nullptr, Error::UNINITIALIZED, "null iterator comparison");
        return _ptr >= rhs._ptr;
    }

    reference operator*() const {
        NDARRAY_ASSERT_CHECK(_ptr != nullptr, Error::UNINITIALIZED, "null iterator dereferenced");
        return *reinterpret_cast<T*>(_ptr);
    }

    pointer operator->() const {
        NDARRAY_ASSERT_CHECK(_ptr != nullptr, Error::UNINITIALIZED, "null iterator dereferenced");
        return reinterpret_cast<T*>(_ptr);
    }

    StridedIterator & operator++() {
        NDARRAY_ASSERT_CHECK(_ptr != nullptr, Error::UNINITIALIZED, "null iterator incremented");
        _ptr += _stride;
        return *this;
    }

    StridedIterator & operator--() {
        NDARRAY_ASSERT_CHECK(_ptr != nullptr, Error::UNINITIALIZED, "null iterator decremented");
        _ptr -= _stride;
        return *this;
    }

    StridedIterator operator++(int) {
        StridedIterator copy(*this);
        ++(*this);
        return copy;
    }

    StridedIterator operator--(int) {
        StridedIterator copy(*this);
        --(*this);
        return copy;
    }

    StridedIterator & operator+=(difference_type n) {
        NDARRAY_ASSERT_CHECK(_ptr != nullptr, Error::UNINITIALIZED, "null iterator advanced");
        _ptr += _stride*n;
        return *this;
    }

    StridedIterator & operator-=(difference_type n) {
        NDARRAY_ASSERT_CHECK(_ptr != nullptr, Error::UNINITIALIZED, "null iterator advanced");
        _ptr -= _stride*n;
        return *this;
    }

    StridedIterator operator+(difference_type n) const {
        return StridedIterator(*this) += n;
    }

    StridedIterator operator-(difference_type n) const {
        return StridedIterator(*this) -= n;
    }

    friend StridedIterator operator+(difference_type n, StridedIterator iter) {
        return iter + n;
    }

    template <typename U>
    difference_type operator-(StridedIterator<U> const & rhs) const {
        NDARRAY_ASSERT_CHECK(_stride == rhs._stride, Error::INCOMPATIBLE_ARGUMENTS,
                             "iterators have different strides");
        NDARRAY_ASSERT_CHECK(_stride != 0, Error::UNINITIALIZED, "iterator stride is zero");
        NDARRAY_ASSERT_CHECK((_ptr - rhs._ptr) % _stride == 0, Error::INCOMPATIBLE_ARGUMENTS,
                             "iterator pointer offset is not a multiple of stride");
        return (_ptr - rhs._ptr)/_stride;
    }

    reference operator[](difference_type n) const {
        NDARRAY_ASSERT_CHECK(_ptr != nullptr, Error::UNINITIALIZED, "null iterator dereferenced");
        return *reinterpret_cast<T*>(_ptr + n*_stride);
    }

private:

    template <typename U> friend class StridedIterator;

    Byte * _ptr;
    Offset _stride;
};

} // ndarray

#endif // !NDARRAY_StridedIterator_hpp_INCLUDED
