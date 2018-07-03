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
#ifndef NDARRAY_detail_ArrayImpl_hpp_INCLUDED
#define NDARRAY_detail_ArrayImpl_hpp_INCLUDED

#include <type_traits>

#include "ndarray/common.hpp"
#include "ndarray/detail/Layout.hpp"

namespace ndarray {
namespace detail {

template <Size N_>
struct ArrayImpl {

    constexpr static Size N = N_;

    ArrayImpl() noexcept = default;

    template <typename Element, typename ShapeVector>
    ArrayImpl(
        ShapeVector const & shape,
        MemoryOrder order,
        TypeTag<Element>
    ) :
        buffer(nullptr),
        layout(Layout<N>::make(shape, sizeof(Element), order))
    {
        std::shared_ptr<Element> tmp(new Element[layout->full_size()], std::default_delete<Element[]>());
        buffer = std::shared_ptr<Byte>(tmp, reinterpret_cast<Byte*>(tmp.get()));
    }

    template <typename Element, typename ShapeVector>
    ArrayImpl(
        std::shared_ptr<Element> const & data,
        ShapeVector const & shape,
        MemoryOrder order
    ) :
        buffer(data, reinterpret_cast<Byte*>(data.get())),
        layout(Layout<N>::make(shape, sizeof(Element), order))
    {}

    template <typename Element, typename ShapeVector, typename StridesVector>
    ArrayImpl(
        std::shared_ptr<Element> const & data,
        ShapeVector const & shape,
        StridesVector const & strides
    ) :
        buffer(data, reinterpret_cast<Byte*>(data.get())),
        layout(Layout<N>::make(shape, strides))
    {}

    template <typename Element>
    ArrayImpl(
        std::shared_ptr<Element> const & data,
        std::shared_ptr<Layout<N> const> layout_
    ) noexcept :
        buffer(data, reinterpret_cast<Byte*>(data.get())),
        layout(std::move(layout_))
    {}

    ArrayImpl(
        std::shared_ptr<Byte> buffer_,
        std::shared_ptr<Layout<N> const> layout_
    ) noexcept :
        buffer(std::move(buffer_)),
        layout(std::move(layout_))
    {}

    ArrayImpl(ArrayImpl const &) noexcept = default;
    ArrayImpl(ArrayImpl &&) noexcept = default;

    ArrayImpl & operator=(ArrayImpl const &) noexcept = default;
    ArrayImpl & operator=(ArrayImpl &&) noexcept = default;

    ~ArrayImpl() noexcept = default;

    template <typename IndexVector>
    Byte * index(IndexVector const & indices) const {
        IndexVectorTraits<IndexVector>::template check_dims<N>(indices);
        Size n = 0;
        Offset offset = 0;
        layout->for_each_dim(
            [&indices, &n, &offset](auto const & d) {
                assert(IndexVectorTraits<IndexVector>::get_size(indices, n) < d.size());
                offset += IndexVectorTraits<IndexVector>::get_size(indices, n)*d.stride();
                ++n;
            }
        );
        return buffer.get() + offset;
    }

    void swap(ArrayImpl & other) noexcept {
        buffer.swap(other);
        layout.swap(other.layout);
    }

    bool operator==(ArrayImpl const & other) const noexcept {
        return buffer == other.buffer &&
            (layout == other.layout
                || (layout && other.layout && *layout == *other.layout));
    }

    bool operator!=(ArrayImpl const & other) const noexcept {
        return !(*this == other);
    }

    std::shared_ptr<Byte> buffer;
    std::shared_ptr<Layout<N> const> layout;
};

template <Size N>
void swap(ArrayImpl<N> & a, ArrayImpl<N> & b) noexcept {
    a.swap(b);
}

} // namespace detail
} // namespace ndarray

#endif // !NDARRAY_detail_ArrayImpl_hpp_INCLUDED
