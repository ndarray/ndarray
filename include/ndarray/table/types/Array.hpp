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
#ifndef NDARRAY_table_types_Array_hpp_INCLUDED
#define NDARRAY_table_types_Array_hpp_INCLUDED

#include <memory>

#include "ndarray/common.hpp"
#include "ndarray/DType.hpp"
#include "ndarray/detail/CompressedPair.hpp"

namespace ndarray {


template <typename T, size_t N>
class DType<Array<T,N,N>> {
public:

    typedef Array<T,N,N> value_type;
    typedef ArrayRef<T,N,N> reference;
    typedef ArrayRef<T const,N,N> const_reference;
    typedef T * pointer;
    typedef T const * const_pointer;
    static constexpr bool is_pod = false;
    static constexpr bool is_direct = true;

    static std::string const & name() {
        // TODO: replace with string formatting
        static std::string const v = type_string<Array<T,N,N>>();
        return v;
    }

    // TODO: user-friendly constructors from shape and strides

    explicit DType(
        std::shared_ptr<detail::Layout> layout,
        DType<T> dtype
    ) : _layout_and_dtype(std::move(layout), std::move(dtype))
    {}

    DType(DType const & other) = default;

    DType(DType && other) = default;

    DType & operator=(DType const & other) = default;

    DType & operator=(DType && other) = default;

    void swap(DType & other) {
        _layout_and_dtype.swap(other._layout_and_dtype);
    }

    size_t alignment() const { return alignof(T); }

    size_t nbytes() const { return sizeof(T)*_layout.full_size(); }

    bool operator==(DType const & other) const {
        return *_layout == *other._layout;
    }

    bool operator!=(DType const & other) const {
        return *_layout != *other._layout;
    }

    reference make_reference_at(
        byte_t * buffer,
        std::shared_ptr<Manager> const & manager
    ) const {
        return reference(
            ArrayImpl<T,N>(buffer, _layout, _nested, manager)
        );
    }

    const_reference make_const_reference_at(
        byte_t const * buffer,
        std::shared_ptr<Manager> const & manager
    ) const {
        return const_reference(
            ArrayImpl<T,N>(
                const_cast<byte_t*>(buffer), _layout, _nested, manager
            )
        );
    }

private:
    detail::CompressedPair<std::shared_ptr<detail::Layout>,DType<T>>
        _layout_and_dtype;
};


} // ndarray

#endif // !NDARRAY_table_types_Array_hpp_INCLUDED