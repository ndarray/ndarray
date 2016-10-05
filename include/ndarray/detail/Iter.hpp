// -*- c++ -*-
/*
 * Copyright (c) 2010-2016, Jim Bosch
 * All rights reserved.
 *
 * ndarray is distributed under a simple BSD-like license;
 * see the LICENSE file that should be present in the root
 * of the source distribution, or alternately available at:
 * https://github.com/ndarray/ndarray
 */
#ifndef NDARRAY_Iter_hpp_INCLUDED
#define NDARRAY_Iter_hpp_INCLUDED

#include "ndarray/common.hpp"
#include "ndarray/detail/ArrayTraits_1.hpp"
#include "ndarray/detail/Array_1.hpp"
#include "ndarray/detail/ArrayRef_1.hpp"
#include "ndarray/detail/IterTraits_1.hpp"

namespace ndarray {

template <typename T>
class Iter {
    typedef detail::IterTraits<T> traits_t;
    typedef typename traits_t::storage storage;
    typedef typename traits_t::offset_ref offset_ref;
    template <typename U> friend class Iter;
public:
    typedef typename traits_t::value_type value_type;
    typedef typename traits_t::reference reference;
    typedef typename traits_t::pointer pointer;
    typedef size_t size_type;
    typedef offset_t difference_type;
    typedef typename traits_t::category iterator_category;

    Iter() : _storage(), _stride() {}

    Iter(storage const & storage_, offset_t stride_) :
        _storage(storage_),
        _stride(stride_)
    {}

    Iter(Iter const & other) = default;

    Iter(Iter && other) = default;

    template <typename U>
    Iter(Iter<U> const & other) :
        _storage(other._storage),
        _stride(other._stride)
    {}

    template <typename U>
    Iter(Iter<U> && other) :
        _storage(std::move(other._storage)),
        _stride(std::move(other._stride))
    {}

    Iter & operator=(Iter const & other) {
        traits_t::reset(_storage, other._storage);
        _stride = other._stride;
        return *this;
    }

    Iter & operator=(Iter && other) {
        traits_t::reset(_storage, std::move(other._storage));
        _stride = std::move(other._stride);
        return *this;
    }

    template <typename U>
    Iter & operator=(Iter<U> const & other) {
        traits_t::reset(_storage, other._storage);
        _stride = other._stride;
        return *this;
    }

    template <typename U>
    Iter & operator=(Iter<U> && other) {
        traits_t::reset(_storage, std::move(other._storage));
        _stride = std::move(other._stride);
        return *this;
    }

    reference operator*() const {
        return traits_t::dereference(_storage);
    }

    reference operator->() const {
        return traits_t::get_pointer(_storage);
    }

    offset_ref operator[](difference_type n) const {
        return traits_t::dereference_at(_storage, n*_stride);
    }

    Iter & operator++() {
        traits_t::advance(_storage, _stride);
        return *this;
    }

    Iter operator++(int) {
        Iter tmp(*this);
        ++(*this);
        return tmp;
    }

    Iter & operator--() {
        traits_t::advance(_storage, -_stride);
        return *this;
    }

    Iter operator--(int) {
        Iter tmp(*this);
        --(*this);
        return tmp;
    }

    Iter & operator+=(difference_type n) {
        traits_t::advance(_storage, n*_stride);
        return *this;
    }

    Iter & operator-=(difference_type n) {
        traits_t::advance(_storage, -n*_stride);
        return *this;
    }

    template <typename U>
    difference_type operator-(Iter<U> & other) const {
        return (traits_t::buffer(_storage)
                - detail::IterTraits<U>::buffer(other._storage))
            / _stride;
    }

    template <typename U>
    bool operator==(Iter<U> const & other) const {
        return traits_t::buffer(_storage)
            == detail::IterTraits<U>::buffer(other._storage);
    }

    template <typename U>
    bool operator!=(Iter<U> const & other) const {
        return traits_t::buffer(_storage)
            != detail::IterTraits<U>::buffer(other._storage);
    }

    template <typename U>
    bool operator<(Iter<U> const & other) const {
        return traits_t::buffer(_storage)
            < detail::IterTraits<U>::buffer(other._storage);
    }

    template <typename U>
    bool operator<=(Iter<U> const & other) const {
        return traits_t::buffer(_storage)
            <= detail::IterTraits<U>::buffer(other._storage);
    }

    template <typename U>
    bool operator>(Iter<U> const & other) const {
        return traits_t::buffer(_storage)
            > detail::IterTraits<U>::buffer(other._storage);
    }

    template <typename U>
    bool operator>=(Iter<U> const & other) const {
        return traits_t::buffer(_storage)
            >= detail::IterTraits<U>::buffer(other._storage);
    }

protected:
    storage _storage;
    offset_t _stride;
};

template <typename T>
Iter<T> operator+(Iter<T> const & it, offset_t n) {
    Iter<T> r(it);
    r += n;
    return r;
}

template <typename T>
Iter<T> operator-(Iter<T> const & it, offset_t n) {
    Iter<T> r(it);
    r -= n;
    return r;
}

template <typename T>
Iter<T> operator+(
    offset_t n,
    Iter<T> const & it
) {
    return it + n;
}

} // namespace ndarray

#endif // !NDARRAY_Iter_hpp_INCLUDED
