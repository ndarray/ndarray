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
#include "ndarray/DType.hpp"
#include "ndarray/Manager.hpp"
#include "ndarray/detail/CompressedPair.hpp"
#include "ndarray/detail/Layout.hpp"

namespace ndarray {
namespace detail {

template <typename T, size_t N>
class ArrayImpl {
public:

    static_assert(
        !std::is_const<T>::value,
        "ArrayImpl should not be instantiated with const types"
    );

    typedef DType<T> dtype_t;
    typedef Layout<N> layout_t;

    ArrayImpl() :
        buffer(nullptr),
        _dtype_and_layout(dtype_t(), std::shared_ptr<Layout<N>>()),
        _manager(nullptr)
    {}

    template <typename ShapeVector>
    ArrayImpl(
        ShapeVector const & shape,
        MemoryOrder order,
        dtype_t const & dtype
    ) :
        buffer(nullptr),
        _dtype_and_layout(dtype, layout_t::make(shape, dtype.nbytes(), order)),
        _manager(nullptr)
    {
        std::tie(buffer, _manager) = manage_new(layout()->full_size(), dtype);
    }

    template <typename ShapeVector, typename StridesVector>
    ArrayImpl(
        byte_t * buffer_,
        ShapeVector const & shape,
        StridesVector const & strides,
        std::shared_ptr<Manager> manager,
        dtype_t const & dtype
    ) :
        buffer(buffer_),
        _dtype_and_layout(dtype, layout_t::make(shape, strides)),
        _manager(std::move(manager))
    {}

    ArrayImpl(
        byte_t * buffer_,
        std::shared_ptr<layout_t> layout_,
        std::shared_ptr<Manager> manager,
        dtype_t const & dtype
    ) :
        buffer(buffer_),
        _dtype_and_layout(dtype, std::move(layout_)),
        _manager(std::move(manager))
    {}

    ArrayImpl(ArrayImpl const &) = default;

    ArrayImpl(ArrayImpl &&) = default;

    ArrayImpl & operator=(ArrayImpl const &) = default;

    ArrayImpl & operator=(ArrayImpl &&) = default;

    void swap(ArrayImpl & other) {
        std::swap(buffer, other.buffer);
        _dtype_and_layout.first().swap(other._dtype_and_layout.first());
        _dtype_and_layout.second().swap(other._dtype_and_layout.second());
        _manager.swap(other._manager);
    }

    byte_t * buffer;

    dtype_t const & dtype() const { return _dtype_and_layout.first(); }

    std::shared_ptr<layout_t> const & layout() const {
        return _dtype_and_layout.second();
    }

    std::shared_ptr<Manager> const & manager() const {
        return _manager;
    }

    bool operator==(ArrayImpl const & other) const {
        return buffer == other.buffer &&
            (layout() == other.layout() || *layout() == *other.layout()) &&
            dtype() == other.dtype();
    }

    bool operator!=(ArrayImpl const & other) const {
        return !(*this == other);
    }

private:
    CompressedPair<dtype_t,std::shared_ptr<layout_t>> _dtype_and_layout;
    std::shared_ptr<Manager> _manager;
};

} // namespace detail
} // namespace ndarray

#endif // !NDARRAY_detail_ArrayImpl_hpp_INCLUDED
