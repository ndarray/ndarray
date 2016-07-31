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
#ifndef NDARRAY_detail_Layout_hpp_INCLUDED
#define NDARRAY_detail_Layout_hpp_INCLUDED

#include <memory>

#include "ndarray/common.hpp"
#include "ndarray/detail/IndexVectorTraits.hpp"
#include "ndarray/Vector.hpp"

namespace ndarray {
namespace detail {

// Storage for the shape and strides of an Array (0-d specialization).
//
// Layout is a recursive container: Layout<N> inherits from Layout<N-1>, with
// 0-d specialized to break the recursion.  Specializing 1-d would also have
// worked, but would require a bit more code duplication.
//
// Layouts are always held by shared_ptr.
//
// The dimensions of Layout are reversed relative to those of an Array:
// for a 3-d Array, the zeroth Array dimension is held in a Layout<3>.
// The get_dim free functions are used to apply this flipping.
template <>
class Layout<0> {
public:

    // Return the size of this dimension (in elements, not bytes).
    constexpr size_t size() const { return 1; }

    // Return the distance between elements in this dimension (in bytes).
    constexpr offset_t stride() const { return 1; }

    // Return the combined size of this and all later (base class) dimensions.
    constexpr size_t full_size() const { return 1; }

    constexpr bool operator==(Layout const & other) const { return true; }
    constexpr bool operator!=(Layout const & other) const { return false; }

protected:

    // Zero shape, zero strides.
    constexpr Layout() {}

    // Explicit shape and strides.
    template <typename ShapeVector, typename StridesVector, size_t M>
    constexpr Layout(
        ShapeVector const * shape,
        StridesVector const * strides,
        std::integral_constant<size_t,M> * full_dim
    ) {}

    // Row-major strides.
    template <typename ShapeVector, size_t M>
    constexpr Layout(
        ShapeVector const * shape,
        size_t nbytes,
        std::integral_constant<size_t,M> * full_dim,
        std::true_type * is_row_major
    ) {}

    // Column-major strides.
    template <typename ShapeVector, size_t M>
    constexpr Layout(
        ShapeVector const * shape,
        offset_t stride,
        std::integral_constant<size_t,M> * full_dim,
        std::false_type * is_row_major
    ) {}

    static constexpr size_t last_stride(size_t nbytes) { return nbytes; }

    template <size_t M>
    void fillShape(Vector<size_t,M> & shape) const {}

    template <size_t M>
    void fillStrides(Vector<offset_t,M> & strides) const {}

};

// Storage for the shape and strides of an Array.
//
// See comments Layout<0> for more information.
template <size_t N>
class Layout : public Layout<N-1> {
public:

    typedef Layout<N-1> base_t;

    // Public factory function from explicit shape and strides.
    template <typename ShapeVector, typename StridesVector>
    static std::shared_ptr<Layout> make(
        ShapeVector const & shape,
        StridesVector const & strides
    ) {
        return std::make_shared<Layout>(shape, strides);
    }

    // Public factory function with explicit shape and automatic strides.
    template <typename ShapeVector>
    static std::shared_ptr<Layout> make(
        ShapeVector const & shape,
        std::size_t nbytes, // sizeof() for a single element
        MemoryOrder order
    ) {
        switch (order) {
            case MemoryOrder::ROW_MAJOR:
                return std::make_shared<Layout>(
                    shape, nbytes,
                    (std::true_type*)nullptr
                );
            case MemoryOrder::COL_MAJOR:
                return std::make_shared<Layout>(
                    shape, nbytes,
                    (std::false_type*)nullptr
                );
            default:
                return nullptr;
        }
    }

    // Zero shape, zero strides.
    Layout() : _size(0), _stride(0) {}

    // Constructor from explicit shape and strides.
    // Should only be called by make().
    template <typename ShapeVector, typename StridesVector>
    Layout(
        ShapeVector const & shape,
        StridesVector const & strides
    ) :
        Layout(&shape, &strides, (std::integral_constant<size_t,N>*)nullptr)
    {}

    // Constructor from shape with automatic row-major strides.
    // Should only be called by make().
    template <typename ShapeVector>
    Layout(
        ShapeVector const & shape,
        std::size_t nbytes, // sizeof() for a single element
        std::true_type* is_row_major
    ) :
        Layout(
            &shape, nbytes, (std::integral_constant<size_t,N>*)nullptr,
            is_row_major
        )
    {}

    // Constructor from shape with automatic col-major strides.
    // Should only be called by make().
    template <typename ShapeVector>
    Layout(
        ShapeVector const & shape,
        std::size_t nbytes, // sizeof() for a single element
        std::false_type* is_row_major
    ) :
        Layout(
            &shape, nbytes, (std::integral_constant<size_t,N>*)nullptr,
            is_row_major
        )
    {}

    Vector<size_t,N> shape() const {
        Vector<size_t,N> result;
        fillShape(result);
        return result;
    }

    Vector<offset_t,N> strides() const {
        Vector<offset_t,N> result;
        fillStrides(result);
        return result;
    }

    // Return the size of this dimension (in elements, not bytes).
    size_t size() const { return _size; }

    // Return the distance between elements in this dimension (in bytes).
    offset_t stride() const { return _stride; }

    // Return the combined size of this and all later (base class) dimensions.
    size_t full_size() const { return _size * base_t::full_size(); }

    bool operator==(Layout const & other) const {
        return _size == other._size && _stride == other._stride
            && base_t::operator==(other);
    }

    bool operator!=(Layout const & other) const {
        return !(*this == other);
    }

protected:

    // Explicit shape and strides, called recursively.
    template <typename ShapeVector, typename StridesVector, size_t M>
    Layout(
        ShapeVector const * shape,
        StridesVector const * strides,
        std::integral_constant<std::size_t,M> * full_dim
    ) :
        base_t(shape, strides, full_dim),
        _size(IndexVectorTraits<ShapeVector>::get_size(*shape, M-N)),
        _stride(IndexVectorTraits<StridesVector>::get_offset(*strides, M-N))
    {}

    // Row-major strides, called recursively.
    template <typename ShapeVector, size_t M>
    Layout(
        ShapeVector const * shape,
        size_t nbytes,
        std::integral_constant<std::size_t,M> * full_dim,
        std::true_type * is_row_major
    ) :
        base_t(shape, nbytes, full_dim, is_row_major),
        _size(IndexVectorTraits<ShapeVector>::get_size(*shape, M-N)),
        _stride(base_t::stride() * base_t::size() * base_t::last_stride(nbytes))
    {}

    // Column-major strides, called recursively.
    template <typename ShapeVector, size_t M>
    Layout(
        ShapeVector const * shape,
        offset_t stride,
        std::integral_constant<std::size_t,M> * full_dim,
        std::false_type * is_row_major
    ) :
        base_t(
            shape,
            stride * IndexVectorTraits<ShapeVector>::get_size(*shape, M-N),
            full_dim,
            is_row_major
        ),
        _size(IndexVectorTraits<ShapeVector>::get_size(*shape, M-N)),
        _stride(stride)
    {}

    static constexpr size_t last_stride(size_t nbytes) { return 1; }

    template <size_t M>
    void fillShape(Vector<size_t,M> & shape) const {
        static_assert(M >= N, "Layout not large enough to fill shape vector.");
        shape[M-N] = _size;
        base_t::fillShape(shape);
    }

    template <size_t M>
    void fillStrides(Vector<offset_t,M> & strides) const {
        static_assert(M >= N, "Layout not large enough to fill strides vector.");
        strides[M-N] = _stride;
        base_t::fillStrides(strides);
    }

private:
    size_t _size;
    offset_t _stride;
};

template <int P, size_t N>
inline Layout<N-P> const &
get_dim(Layout<N> const & layout) { return layout; }

template <int P, size_t N>
inline std::shared_ptr<Layout<N-P>>
get_dim(std::shared_ptr<Layout<N>> const & layout) { return layout; }

} // namespace detail
} // ndarray

#endif // !NDARRAY_detail_Layout_hpp_INCLUDED
