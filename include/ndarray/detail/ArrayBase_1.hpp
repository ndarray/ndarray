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
#ifndef NDARRAY_ArrayBase_1_hpp_INCLUDED
#define NDARRAY_ArrayBase_1_hpp_INCLUDED

#include "ndarray/common.hpp"
#include "ndarray/detail/ArrayTraits_1.hpp"
#include "ndarray/detail/ArrayImpl.hpp"

namespace ndarray {

template <typename T, int N_>
class ArrayBase {

    typedef detail::ArrayTraits<T,N_> traits_t;
    typedef typename traits_t::impl_t impl_t;
    typedef typename traits_t::layout_t layout_t;

    template <typename U, int M> friend class Array;
    template <typename U, int M> friend class ArrayRef;
    template <typename U, int M> friend struct detail::IterImpl;
    template <typename U, int M> friend struct detail::ArrayTraits;

    std::shared_ptr<layout_t> const & _layout() const { return _impl.layout(); }

public:

    typedef T element;
    static constexpr int N = N_;
    typedef typename traits_t::dtype_t dtype_t;
    typedef typename traits_t::value_type value_type;
    typedef typename traits_t::reference reference;
    typedef typename traits_t::pointer pointer;
    typedef typename traits_t::iterator iterator;

    iterator begin() const;

    iterator end() const;

    reference operator[](size_t n) const;

    element * data() const { return reinterpret_cast<T*>(_impl.buffer); }

    bool empty() const { return !_layout(); }

    size_t size() const { return _layout()->size(); }

    offset_t stride() const { return _layout()->stride(); }

    template <int M>
    size_t size() const { return detail::get_dim<M>(*_layout()).size(); }

    template <int M>
    offset_t stride() const { return detail::get_dim<M>(*_layout()).stride(); }

    size_t full_size() const { return _layout()->full_size(); }

    dtype_t const & dtype() const { return _impl.dtype(); }

    std::shared_ptr<Manager> const & manager() const { return _impl.manager(); }

    bool operator==(ArrayBase const & other) const {
        return _impl == other._impl;
    }

    bool operator!=(ArrayBase const & other) const {
        return _impl != other._impl;
    }

    void swap(ArrayBase & other) {
        _impl.swap(other._impl);
    }

protected:

    ArrayBase() : _impl() {}

    explicit ArrayBase(impl_t const & impl) : _impl(impl) {}

    explicit ArrayBase(impl_t && impl) : _impl(std::move(impl)) {}

    ArrayBase(ArrayBase const &) = default;

    ArrayBase(ArrayBase &&) = default;

    ArrayBase & operator=(ArrayBase const &) = default;

    ArrayBase & operator=(ArrayBase &&) = default;

private:
    impl_t _impl;
};


template <int M, typename T, int N>
inline size_t get_size(ArrayBase<T,N> const & x) {
    return x.template size<M>();
}

template <int M, typename T, int N>
inline offset_t get_stride(ArrayBase<T,N> const & x) {
    return x.template stride<M>();
}


} // namespace ndarray

#endif // !NDARRAY_ArrayBase_1_hpp_INCLUDED
