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
#ifndef NDARRAY_Iter_2_hpp_INCLUDED
#define NDARRAY_Iter_2_hpp_INCLUDED

#include "ndarray/common.hpp"
#include "ndarray/detail/ArrayImpl.hpp"
#include "ndarray/detail/ArrayBase_1.hpp"
#include "ndarray/detail/Array_1.hpp"
#include "ndarray/detail/ArrayRef_1.hpp"
#include "ndarray/detail/IterImpl.hpp"
#include "ndarray/detail/Iter_1.hpp"
#include "ndarray/detail/IterTraits_2.hpp"

namespace ndarray {

template <typename T, size_t N, offset_t C>
typename detail::IterTraits<T const,N,C>::actual_ref
Iter<T const,N,C>::operator*() const {
    return traits_t::make_reference(*this);
}

template <typename T, size_t N, offset_t C>
typename detail::IterTraits<T const,N,C>::actual_ptr
Iter<T const,N,C>::operator->() const {
    return traits_t::make_pointer(*this);
}

template <typename T, size_t N, offset_t C>
typename detail::IterTraits<T const,N,C>::reference
Iter<T const,N,C>::operator[](difference_type n) const {
    return traits_t::make_reference_at(
        _impl.buffer() + n*_impl.stride(),
        *this
    );
}

template <typename T, size_t N, offset_t C>
typename detail::IterTraits<T,N,C>::actual_ref Iter<T,N,C>::operator*() const {
    return traits_t::make_reference(*this);
}

template <typename T, size_t N, offset_t C>
typename detail::IterTraits<T,N,C>::actual_ptr Iter<T,N,C>::operator->() const {
    return traits_t::make_pointer(*this);
}

template <typename T, size_t N, offset_t C>
typename detail::IterTraits<T,N,C>::reference
Iter<T,N,C>::operator[](difference_type n) const {
    return traits_t::make_reference_at(
        this->_impl.buffer() + n*this->_impl.stride(),
        *this
    );
}

} // namespace ndarray

#endif // !NDARRAY_Iter_2_hpp_INCLUDED
