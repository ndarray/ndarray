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
#ifndef NDARRAY_detail_layout_hpp_INCLUDED
#define NDARRAY_detail_layout_hpp_INCLUDED

#include <memory>

#include "ndarray/common.hpp"
#include "ndarray/detail/index_vector_traits.hpp"

namespace ndarray {
namespace detail {

template <>
class layout<0> {
public:

    constexpr size_t size() const { return 1; }

    constexpr offset_t stride() const { return 1; }

    constexpr size_t full_size() const { return 1; }

    template <typename ShapeVector, typename StridesVector>
    layout(ShapeVector const * shape, StridesVector const * strides) {}

    // Row-major strides
    template <typename ShapeVector>
    layout(ShapeVector const * shape, size_t nbytes) {}

    // Column-major strides
    template <typename ShapeVector>
    layout(ShapeVector const * shape, size_t nbytes, offset_t stride) {}

    // Zero shape and strides
    layout() {}

};

template <int N>
class layout : public layout<N-1> {
public:

    typedef layout<N-1> base_t;

    size_t size() const { return _size; }

    offset_t stride() const { return _stride; }

    size_t full_size() const { return _size * base_t::full_size(); }

    template <typename ShapeVector, typename StridesVector>
    static std::shared_ptr<layout> make(ShapeVector const & shape, StridesVector const & strides) {
        return std::make_shared<layout>(&shape, &strides);
    }

    template <typename ShapeVector, typename StridesVector>
    static std::shared_ptr<layout> make(ShapeVector const & shape, std::size_t nbytes, MemoryOrder order) {
        switch (order) {
            case ROW_MAJOR:
                return std::make_shared<layout>(&shape, nbytes);
            case COL_MAJOR:
                return std::make_shared<layout>(&shape, nbytes, 1);
        }
    }

    template <typename ShapeVector, typename StridesVector>
    layout(ShapeVector const * shape, StridesVector const * strides) :
        base_t(shape, strides),
        _size(index_vector_traits::get_size(*shape, M-N)),
        _stride(index_vector_traits::get_offset(*shape, M-N))
    {}

    // Row-major strides
    template <typename ShapeVector>
    layout(ShapeVector const * shape, size_t nbytes) :
        base_t(shape),
        _size(index_vector_traits::get_size(*shape, M-N)),
        _stride(base_t::stride() * base_t::size() * nbytes)
    {}

    // Column-major strides
    template <typename ShapeVector>
    layout(ShapeVector const * shape, size_t nbytes, offset_t stride) :
        base_t(shape, nbytes, stride * index_vector_traits::get_size(*shape, M-N)),
        _size(index_vector_traits::get_size(*shape, M-N)),
        _stride(stride)
    {}

    // Zero shape and strides
    layout() : _size(0), _stride(0) {}

private:
    size_t _size;
    offset_t _stride;
};

template <int P, int N>
inline layout<N-P> const &
get_dim(layout<N> const & layout) { return layout; }

template <int P, int N>
inline std::shared_ptr<layout<N-P>>
get_dim(std::shared_ptr<layout<N>> const & layout) { return layout; }

} // namespace detail
} // ndarray

#endif // !NDARRAY_detail_layout_hpp_INCLUDED
