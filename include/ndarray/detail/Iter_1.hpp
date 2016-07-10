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
#ifndef NDARRAY_Iter_1_hpp_INCLUDED
#define NDARRAY_Iter_1_hpp_INCLUDED

#include "ndarray/common.hpp"
#include "ndarray/detail/ArrayTraits_1.hpp"
#include "ndarray/detail/Array_1.hpp"
#include "ndarray/detail/ArrayRef_1.hpp"
#include "ndarray/detail/IterTraits_1.hpp"
#include "ndarray/detail/IterImpl.hpp"

namespace ndarray {

template <typename T, int N>
class Iter<T const,N> {
    typedef detail::IterTraits<T const,N> traits_t;
    typedef typename traits_t::impl_t impl_t;
    template <typename U, int M> friend class detail::IterTraits;
public:
    typedef typename traits_t::value_type value_type;
    typedef typename traits_t::reference reference;
    typedef typename traits_t::pointer pointer;
    typedef size_t size_type;
    typedef offset_t difference_type;
    typedef typename traits_t::category iterator_category;
    typedef typename traits_t::dtype_t dtype_t;

    Iter() : _impl() {}

    Iter(Iter const & other) = default;

    Iter(Iter && other) = default;

    Iter & operator=(Iter const & other) = default;

    Iter & operator=(Iter && other) = default;

    typename traits_t::actual_ref operator*() const;

    typename traits_t::actual_ptr operator->() const;

    reference operator[](difference_type n) const;

    Iter & operator++() {
        _impl.advance(1);
        return *this;
    }

    Iter operator++(int) {
        Iter tmp(*this);
        ++(*this);
        return tmp;
    }

    Iter & operator--() {
        _impl.advance(-1);
    }

    Iter operator--(int) {
        Iter tmp(*this);
        --(*this);
        return tmp;
    }

    Iter & operator+=(difference_type n) {
        _impl.advance(n);
        return *this;
    }

    Iter & operator-=(difference_type n) {
        _impl.advance(-n);
        return *this;
    }

    difference_type operator-(Iter & other) const {
        return (_impl.buffer() - other.buffer()) / _impl.stride();
    }

    bool operator==(Iter const & other) const {
        return _impl.buffer() == other._impl.buffer();
    }

    bool operator!=(Iter const & other) const {
        return _impl.buffer() != other._impl.buffer();
    }

    bool operator<(Iter const & other) const {
        return _impl.buffer() < other._impl.buffer();
    }

    bool operator<=(Iter const & other) const {
        return _impl.buffer() <= other._impl.buffer();
    }

    bool operator>(Iter const & other) const {
        return _impl.buffer() > other._impl.buffer();
    }

    bool operator>=(Iter const & other) const {
        return _impl.buffer() >= other._impl.buffer();
    }

protected:

    explicit Iter(impl_t const & impl) : _impl(impl) {}

    explicit Iter(impl_t && impl) : _impl(std::move(impl)) {}

    impl_t _impl;
};

template <typename T, int N>
Iter<T const,N> operator+(Iter<T const,N> const & it, offset_t n) {
    Iter<T const,N> r(it);
    r += n;
    return r;
}

template <typename T, int N>
Iter<T const,N> operator-(Iter<T const,N> const & it, offset_t n) {
    Iter<T const,N> r(it);
    r -= n;
    return r;
}

template <typename T, int N>
Iter<T const,N> operator+(offset_t n, Iter<T const,N> const & it) {
    return it + n;
}

template <typename T, int N>
class Iter : public Iter<T const,N> {
    typedef detail::IterTraits<T,N> traits_t;
    typedef typename traits_t::impl_t impl_t;
    typedef Iter<T const,N> base_t;
    template <typename U, int M> friend class detail::IterTraits;
public:
    typedef typename traits_t::value_type value_type;
    typedef typename traits_t::reference reference;
    typedef typename traits_t::pointer pointer;
    typedef size_t size_type;
    typedef offset_t difference_type;
    typedef typename traits_t::category iterator_category;
    typedef typename traits_t::dtype_t dtype_t;

    Iter() : base_t() {}

    Iter(Iter const & other) = default;

    Iter(Iter && other) = default;

    Iter & operator=(Iter const & other) = default;

    Iter & operator=(Iter && other) = default;

    typename traits_t::actual_ref operator*() const;

    typename traits_t::actual_ptr operator->() const;

    reference operator[](difference_type n) const;

    Iter & operator++() {
        this->_impl.advance(1);
        return *this;
    }

    Iter operator++(int) {
        Iter tmp(*this);
        ++(*this);
        return tmp;
    }

    Iter& operator--() {
        this->_impl.advance(-1);
    }

    Iter operator--(int) {
        Iter tmp(*this);
        --(*this);
        return tmp;
    }

    Iter & operator+=(difference_type n) {
        this->_impl.advance(n);
        return *this;
    }

    Iter & operator-=(difference_type n) {
        this->_impl.advance(-n);
        return *this;
    }

private:

    explicit Iter(impl_t const & impl) : base_t(impl) {}

    explicit Iter(impl_t && impl) : base_t(std::move(impl)) {}

};

template <typename T, int N>
Iter<T,N> operator+(Iter<T,N> const & it, offset_t n) {
    Iter<T,N> r(it);
    r += n;
    return r;
}

template <typename T, int N>
Iter<T,N> operator-(Iter<T,N> const & it, offset_t n) {
    Iter<T,N> r(it);
    r -= n;
    return r;
}

template <typename T, int N>
Iter<T,N> operator+(offset_t n, Iter<T,N> const & it) {
    return it + n;
}

} // namespace ndarray

#endif // !NDARRAY_Iter_1_hpp_INCLUDED
