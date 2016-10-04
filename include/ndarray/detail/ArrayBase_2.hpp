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
#ifndef NDARRAY_detail_ArrayBase_2_hpp_INCLUDED
#define NDARRAY_detail_ArrayBase_2_hpp_INCLUDED

#include "ndarray/common.hpp"
#include "ndarray/detail/ArrayRef_1.hpp"
#include "ndarray/detail/Iter.hpp"
#include "ndarray/detail/IterTraits_2.hpp"
#include "ndarray/detail/ArrayTraits_2.hpp"
#include "ndarray/detail/IndexVectorTraits.hpp"
#include "ndarray/detail/Layout.hpp"

namespace ndarray {

template <typename T, size_t N, offset_t C>
inline auto ArrayBase<T,N,C>::begin() const -> iterator {
    return detail::ArrayTraits<T,N,C>::make_iterator_at(
        this->_impl.buffer,
        *this
    );
}

template <typename T, size_t N, offset_t C>
inline auto ArrayBase<T,N,C>::end() const -> iterator {
    return detail::ArrayTraits<T,N,C>::make_iterator_at(
        this->_impl.buffer + this->stride()*this->size(),
        *this
    );
}

template <typename T, size_t N, offset_t C>
inline auto ArrayBase<T,N,C>::operator[](size_t n) const -> reference {
    return traits_t::make_reference_at(
        this->_impl.buffer + this->stride()*n,
        *this
    );
}

template <typename T, size_t N, offset_t C>
template <typename IndexVector>
inline auto ArrayBase<T,N,C>::_at(IndexVector const & index) const -> element {
    detail::IndexVectorTraits<IndexVector>::template check_dims<N>(index);
    detail::StrideInnerProduct<IndexVector> func(index);
    this->_layout()->for_each_dim(func);
    return this->dtype().make_reference_at(
        this->_impl.buffer + func.offset,
        this->manager()
    );
}

} // namespace ndarray

#endif // !NDARRAY_detail_ArrayBase_2_hpp_INCLUDED
