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
#ifndef NDARRAY_Vector_hpp_INCLUDED
#define NDARRAY_Vector_hpp_INCLUDED

#include <numeric>
#include <array>

#include "ndarray/common.hpp"

namespace ndarray {

template <typename T, size_t N>
class Vector {
    typedef std::array<T,N> impl_t;
public:
    typedef typename impl_t::value_type value_type;
    typedef typename impl_t::size_type size_type;
    typedef typename impl_t::difference_type difference_type;
    typedef typename impl_t::reference reference;
    typedef typename impl_t::const_reference const_reference;
    typedef typename impl_t::pointer pointer;
    typedef typename impl_t::const_pointer const_pointer;
    typedef typename impl_t::iterator iterator;
    typedef typename impl_t::const_iterator const_iterator;
    typedef typename impl_t::reverse_iterator reverse_iterator;
    typedef typename impl_t::const_reverse_iterator const_reverse_iterator;

    explicit Vector(T value = 0) : _impl() {
        _impl.fill(value);
    }

    explicit Vector(std::initializer_list<T> other) : _impl() {
        assert(other.size() == N);
        std::copy(other.begin(), other.end(), begin());
    }

    Vector(Vector const &) = default;
    Vector(Vector &&) = default;

    Vector & operator=(Vector const &) = default;
    Vector & operator=(Vector &&) = default;

    void swap(Vector & other) { _impl.swap(other._impl); }

    reference operator[](size_type n) { return _impl[n]; }
    const_reference operator[](size_type n) const { return _impl[n]; }

    reference front() { return _impl.front(); }
    const_reference front() const { return _impl.front(); }

    reference back() { return _impl.back(); }
    const_reference back() const { return _impl.back(); }

    pointer data() { return _impl.data(); }
    const_pointer data() const { return _impl.data(); }

    iterator begin() { return _impl.begin(); }
    const_iterator begin() const { return _impl.cbegin(); }
    const_iterator cbegin() const { return _impl.cbegin(); }

    iterator end() { return _impl.end(); }
    const_iterator end() const { return _impl.cend(); }
    const_iterator cend() const { return _impl.cend(); }

    reverse_iterator rbegin() { return _impl.rbegin(); }
    const_reverse_iterator rbegin() const { return _impl.crbegin(); }
    const_reverse_iterator crbegin() const { return _impl.crbegin(); }

    reverse_iterator rend() { return _impl.rend(); }
    const_reverse_iterator rend() const { return _impl.crend(); }
    const_reverse_iterator crend() const { return _impl.crend(); }

    constexpr bool empty() const { return N == 0u; }
    constexpr size_type size() const { return N; }
    constexpr size_type max_size() const { return N; }

    bool equals(std::initializer_list<T> other) const {
        assert(N == other.size());
        return std::equal(cbegin(), cend(), other.begin());
    }

    value_type dot(Vector const & other) const {
        return std::inner_product(
            cbegin(), cend(),
            other.cbegin(), other.cend(),
            0
        );
    }

    Vector reversed() const {
        Vector result;
        std::copy(crbegin(), crend(), result.begin());
        return result;
    }

private:
    impl_t _impl;
};


} // namespace ndarray

#endif // !NDARRAY_Vector_hpp_INCLUDED
