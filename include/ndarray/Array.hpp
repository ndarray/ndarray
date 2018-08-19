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
#ifndef NDARRAY_Array_hpp_INCLUDED
#define NDARRAY_Array_hpp_INCLUDED

#include "ndarray/common.hpp"
#include "ndarray/IndexVectorTraits.hpp"
#include "ndarray/ArrayInterfaceN.hpp"
#include "ndarray/detail/ArrayImpl.hpp"

namespace ndarray {

namespace detail {

constexpr bool contiguousness_convertible(Size n, Offset in, Offset out) {
    return (in >= 0 && out >= 0 && in >= out) ||
           (in <= 0 && out <= 0 && in <= out) ||
           (n == 1 && (in == 1 || in == -1) && (out == 1 || out == -1));
}

} // namespace detail


template <typename T, Size N, Offset C>
class Array<T const, N, C> : public ArrayInterfaceN<Array<T const, N, C>, T const, N, C> {
public:

    Array() noexcept = default;

    Array(Array const &) noexcept = default;

    Array(Array &&) noexcept = default;

    template <Offset D>
    Array(Array<T const, N, D> const & other) noexcept : _impl(other._impl) {
        static_assert(detail::contiguousness_convertible(N, D, C), "invalid contiguousness conversion");
    }

    template <Offset D>
    Array(Array<T const, N, D> && other) noexcept : _impl(other._impl) {
        static_assert(detail::contiguousness_convertible(N, D, C), "invalid contiguousness conversion");
    }

    template <typename ShapeVector>
    explicit Array(
        ShapeVector const & shape,
        EnableIfIndexVector<MemoryOrder, ShapeVector> order=MemoryOrder::ROW_MAJOR
    ) : _impl(shape, order, detail::TypeTag<T>()) {}

    explicit Array(std::initializer_list<Size> shape, MemoryOrder order=MemoryOrder::ROW_MAJOR) :
        _impl(shape, order, detail::TypeTag<T>()) {}

    template <typename ShapeVector>
    Array(
        std::shared_ptr<T const> data,
        ShapeVector const & shape,
        EnableIfIndexVector<MemoryOrder, ShapeVector> order=MemoryOrder::ROW_MAJOR
    ) : _impl(std::const_pointer_cast<T>(std::move(data)), shape, order) {}

    Array(std::shared_ptr<T const> data, std::initializer_list<Size> shape,
          MemoryOrder order=MemoryOrder::ROW_MAJOR) :
        _impl(std::const_pointer_cast<T>(std::move(data)), shape, order) {}

    template <typename ShapeVector, typename StridesVector>
    Array(
        EnableIfIndexVector<std::shared_ptr<T const>, ShapeVector, StridesVector> data,
        ShapeVector const & shape,
        StridesVector const & strides
    ) : _impl(std::const_pointer_cast<T>(std::move(data)), shape, strides) {
        _impl.layout->template check_contiguousness<C>(sizeof(T));
    }

    template <typename ShapeVector, typename StridesVector>
    Array(
        std::shared_ptr<T const> data,
        std::initializer_list<Size> shape,
        std::initializer_list<Offset> strides
    ) : _impl(std::const_pointer_cast<T>(std::move(data)), shape, strides) {
        _impl.layout->template check_contiguousness<C>(sizeof(T));
    }

    Array(std::shared_ptr<Byte const> buffer, std::shared_ptr<detail::Layout<N> const> layout) :
        _impl(std::const_pointer_cast<Byte>(std::move(buffer)), std::move(layout))
    {
        _impl.layout->template check_contiguousness<C>(sizeof(T));
    }

    Array & operator=(Array const &) noexcept = default;

    Array & operator=(Array &&) noexcept = default;

    Deref<Array<T const, N, C>> operator*() const noexcept;

    bool empty() const { return (!_impl.layout) || _impl.layout->size() == 0u; }

    Size size() const { return _impl.layout ? _impl.layout->size() : 0u; }

    Offset stride() const { return _impl.layout->stride(); }

    std::array<Size, N> shape() const { return _impl.layout->shape(); }

    std::array<Offset, N> strides() const { return _impl.layout->strides(); }

    Size full_size() const { return _impl.layout->full_size(); }

protected:

    template <typename T2, Size N2, Offset C2> friend class Array;
    template <typename Derived, typename T2, Size N2, Offset C2> friend class ArrayInterfaceN;

    detail::ArrayImpl<N> _impl;
};


template <typename T, Size N, Offset C>
class Array : public ArrayInterfaceN<Array<T, N, C>, T, N, C>, public Array<T const, N, C> {
    using Base = Array<T const, N, C>;
public:

    Array() noexcept = default;

    Array(Array const &) noexcept = default;

    Array(Array &&) noexcept = default;

    template <Offset D>
    Array(Array<T, N, D> const & other) noexcept : Base(other) {}

    template <Offset D>
    Array(Array<T, N, D> && other) noexcept : Base(other) {}

    template <typename ShapeVector>
    explicit Array(
        ShapeVector const & shape,
        EnableIfIndexVector<MemoryOrder, ShapeVector> order=MemoryOrder::ROW_MAJOR
    ) : Base(shape, order) {}

    explicit Array(std::initializer_list<Size> shape, MemoryOrder order=MemoryOrder::ROW_MAJOR) :
        Base(shape, order) {}

    template <typename ShapeVector>
    Array(
        std::shared_ptr<T> data,
        ShapeVector const & shape,
        EnableIfIndexVector<MemoryOrder, ShapeVector> order=MemoryOrder::ROW_MAJOR
    ) : Base(std::move(data), shape, order) {}

    Array(std::shared_ptr<T> data, std::initializer_list<Size> shape,
          MemoryOrder order=MemoryOrder::ROW_MAJOR) :
        Base(std::move(data), shape, order) {}

    template <typename ShapeVector, typename StridesVector>
    Array(
        EnableIfIndexVector<std::shared_ptr<T>, ShapeVector, StridesVector> data,
        ShapeVector const & shape,
        StridesVector const & strides
    ) : Base(std::move(data), shape, strides) {}

    template <typename ShapeVector, typename StridesVector>
    Array(
        std::shared_ptr<T> data,
        std::initializer_list<Size> shape,
        std::initializer_list<Offset> strides
    ) : Base(std::move(data), shape, strides) {}

    Array(std::shared_ptr<Byte> buffer, std::shared_ptr<detail::Layout<N> const> layout) :
        Base(std::move(buffer), std::move(layout)) {}

    Array & operator=(Array const &) noexcept = default;

    Array & operator=(Array &&) noexcept = default;

    Deref<Array<T, N, C>> operator*() const noexcept;

};

} // namespace ndarray

#endif // !NDARRAY_Array_hpp_INCLUDED
