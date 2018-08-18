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
#ifndef NDARRAY_detail_Layout_hpp_INCLUDED
#define NDARRAY_detail_Layout_hpp_INCLUDED

#include <memory>

#include "ndarray/common.hpp"
#include "ndarray/exceptions.hpp"
#include "ndarray/IndexVectorTraits.hpp"

namespace ndarray {
namespace detail {

template <> class Layout<0>;

// Storage for the shape and strides of an Array.
//
// Layout is a recursive container: Layout<N> inherits from Layout<N-1>, with
// 0-d specialized to break the recursion.  Specializing 1-d would also have
// worked, but would require a bit more code duplication.
//
// Layouts are always held by shared_ptr.
//
// The dimensions of Layout are reversed relative to those of an Array:
// for a 3-d Array, the zeroth (outermost) Array dimension is held in a
// Layout<3>, while the last (innermost) dimension is held in a Layout<1>.
// The get_dim free functions can be used to obtain the Layout specialization
// for a particular dimension.
template <Size N>
class Layout : public Layout<N-1> {
public:

    using Base = Layout<N-1>;

    // Public factory function from explicit shape and strides.
    template <typename ShapeVector, typename StridesVector>
    static std::shared_ptr<Layout const> make(ShapeVector const & shape, StridesVector const & strides) {
        return std::make_shared<ConstructionHelper>(shape, strides);
    }

    // Public factory function with explicit shape and automatic strides.
    template <typename ShapeVector>
    static std::shared_ptr<Layout const> make(ShapeVector const & shape, Size element_size, MemoryOrder order) {
        switch (order) {
        case MemoryOrder::ROW_MAJOR:
            return std::make_shared<ConstructionHelper>(shape, element_size, RowMajorTag());
        case MemoryOrder::COL_MAJOR:
            return std::make_shared<ConstructionHelper>(shape, element_size, ColMajorTag());
        default:
            return nullptr;
        }
    }

    // Since Layouts are always passed/held by shared_ptr, an attempt to copy
    // or move one is a mistake we'd like to catch.
    Layout(Layout const &) = delete;
    Layout(Layout &&) = delete;

    Layout & operator=(Layout const &) = delete;
    Layout & operator=(Layout &&) = delete;

    ~Layout() noexcept = default;

    std::array<Size, N> shape() const noexcept {
        std::array<Size, N> result;
        fill_shape(result);
        return result;
    }

    std::array<Offset, N> strides() const noexcept {
        std::array<Offset, N> result;
        fill_strides(result);
        return result;
    }

    // Return the size of this dimension (in elements, not bytes).
    Size size() const noexcept { return _size; }

    // Return the distance between elements in this dimension (in bytes).
    Offset stride() const noexcept { return _stride; }

    // Return the combined size of this and all later (base class) dimensions.
    Size full_size() const noexcept { return _size * Base::full_size(); }

    bool operator==(Layout const & other) const noexcept {
        return _size == other._size && _stride == other._stride
            && Base::operator==(other);
    }

    bool operator!=(Layout const & other) const noexcept {
        return !(*this == other);
    }

    // Call the given function on this Layout and all nonzero base classes,
    // starting with the outermost dimension (Layout<N>) and ending with the
    // innermost (Layout<1>).
    template <typename F>
    void for_each_dim(F func) const {
        func(*this);
        Base::for_each_dim(func);
    }

    // Call the given function on this Layout and all nonzero base classes,
    // starting with the innermost dimension (Layout<0>) and ending with
    // the outermost (Layout<N>).
    template <typename F>
    void for_each_dim_r(F func) const {
        Base::for_each_dim_r(func);
        func(*this);
    }

    Size count_contiguous_dims(Size element_size, MemoryOrder order) const noexcept {
        Size n_contiguous_dims = 0;
        Offset contiguous_stride = element_size;
        auto func = [&n_contiguous_dims, &contiguous_stride](auto const & layout) -> bool {
            if (contiguous_stride == layout.stride()) {
                ++n_contiguous_dims;
                contiguous_stride *= layout.size();
            } else {
                contiguous_stride = 0;  // make sure the test never succeeds for any future dimension
            }
        };
        switch (order) {
        case MemoryOrder::ROW_MAJOR:
            for_each_dim_r(func);
            break;
        case MemoryOrder::COL_MAJOR:
            for_each_dim(func);
            break;
        };
        return n_contiguous_dims;
    }

    // Throw NoncontiguousError if this does not have at least C row-major
    // contiguous dimensions (starting from the innermost) or, if C is negative
    // at least -C column-major contiguous dimensions (starting from the
    // outermost).
    template <Offset C>
    void check_contiguousness(Size element_size) const {
        static_assert(
            static_cast<Offset>(N) >= C && -static_cast<Offset>(N) <= C,
            "Cannot have more contiguous dimensions than total dimensions."
        );
        if (C == 0) return;
        Size n_contiguous_dims = count_contiguous_dims(
            element_size,
            C > 0 ? MemoryOrder::ROW_MAJOR : MemoryOrder::COL_MAJOR
        );
        if (std::abs(C) > n_contiguous_dims) {
            throw NoncontiguousError(n_contiguous_dims, C);
        }
    }

protected:

    // Explicit shape and strides, called recursively.
    template <typename ShapeVector, typename StridesVector, Size M>
    Layout(
        ShapeVector const & shape,
        StridesVector const & strides,
        std::integral_constant<Size, M>
    ) :
        Base(shape, strides, std::integral_constant<Size, M>()),
        _size(IndexVectorTraits<ShapeVector>::get_size(shape, M-N)),
        _stride(IndexVectorTraits<StridesVector>::get_offset(strides, M-N))
    {}

    // Compute row-major strides, called recursively.
    template <typename ShapeVector, Size M>
    Layout(
        ShapeVector const & shape,
        Size element_size,
        std::integral_constant<Size, M>,
        RowMajorTag
    ) :
        Base(shape, element_size, std::integral_constant<Size, M>(), RowMajorTag()),
        _size(IndexVectorTraits<ShapeVector>::get_size(shape, M-N)),
        _stride(Base::stride() * Base::size() * Base::last_stride(element_size))
    {}

    // Compute column-major strides, called recursively.
    template <typename ShapeVector, Size M>
    Layout(
        ShapeVector const & shape,
        Offset stride,
        std::integral_constant<Size, M>,
        ColMajorTag
    ) :
        Base(
            shape,
            stride * IndexVectorTraits<ShapeVector>::get_size(shape, M-N),
            std::integral_constant<Size, M>(),
            ColMajorTag()
        ),
        _size(IndexVectorTraits<ShapeVector>::get_size(shape, M-N)),
        _stride(stride)
    {}

    static Size last_stride(Size element_size) noexcept { return 1; }

    // Write the full array shape into the given array.
    template <Size M>
    void fill_shape(std::array<Size, M> & shape) const {
        static_assert(M >= N, "Layout not large enough to fill shape vector.");
        shape[M-N] = _size;
        Base::fill_shape(shape);
    }

    // Write strides for all dimensions into the given array.
    template <Size M>
    void fill_strides(std::array<Offset, M> & strides) const {
        static_assert(M >= N, "Layout not large enough to fill strides vector.");
        strides[M-N] = _stride;
        Base::fill_strides(strides);
    }

private:

    // Private inner class that inherits from Layout and has public ctors.
    // This lets us use make_shared in Layout::make without making Layout's
    // own constructors public.
    class ConstructionHelper;

    Size _size;
    Offset _stride;
};


// Trivial private subclass of Layout, with public ctors usable by make_shared.
template <Size N>
class Layout<N>::ConstructionHelper : public Layout<N> {
public:

    // Constructor from explicit shape and strides.
    // Only called by Layout::make().
    template <typename ShapeVector, typename StridesVector>
    ConstructionHelper(ShapeVector const & shape, StridesVector const & strides) :
        Layout<N>(shape, strides, std::integral_constant<Size, N>())
    {}

    // Constructor from shape and automatic strides.
    // Only called by Layout::make().
    template <typename ShapeVector, typename Tag>
    ConstructionHelper(ShapeVector const & shape, Size element_size, Tag) :
        Layout<N>(shape, element_size, std::integral_constant<Size, N>(), Tag())
    {}

};


// Empty 0-d specialization of Layout to break template recursion.
template <>
class Layout<0> {
public:

    Layout(Layout const &) = delete;
    Layout(Layout &&) = delete;
    Layout & operator=(Layout const &) = delete;
    Layout & operator=(Layout &&) = delete;

    ~Layout() noexcept = default;

    // Return the size of this dimension (in elements, not bytes).
    Size size() const noexcept { return 1; }

    // Return the distance between elements in this dimension (in bytes).
    Offset stride() const noexcept { return 1; }

    // Return the combined size of this and all later (base class) dimensions.
    Size full_size() const noexcept { return 1; }

    bool operator==(Layout const & other) const noexcept { return true; }
    bool operator!=(Layout const & other) const noexcept { return false; }

    template <typename F>
    bool for_each_dim(F) const noexcept { return true; }

    template <typename F>
    bool for_each_dim_r(F) const noexcept { return true; }

protected:

    // Explicit shape and strides.  Only called by Layout<1>.
    template <typename ShapeVector, typename StridesVector, Size M>
    Layout(
        ShapeVector const & shape,
        StridesVector const & strides,
        std::integral_constant<Size, M>
    ) noexcept {}

    // Automatic strides.  Only called by Layout<1>.
    template <typename ShapeVector, Size M, typename Tag>
    Layout(
        ShapeVector const & shape,
        Size element_size,
        std::integral_constant<Size, M>,
        Tag
    ) noexcept {}

    static Size last_stride(Size element_size) noexcept { return element_size; }

    template <Size M>
    void fill_shape(std::array<Size, M> & shape) const noexcept {}

    template <Size M>
    void fill_strides(std::array<Offset, M> & strides) const noexcept {}

};


// Retrieve the Layout dimension specialization corresponding to the Pth
// dimension out of N total dimensions.
template <Size P, Size N>
inline Layout<N-P> const &
get_dim(Layout<N> const & layout) { return layout; }


// Retrieve the Layout dimension specialization corresponding to the Pth
// dimension out of N total dimensions.
template <Size P, Size N>
inline std::shared_ptr<Layout<N-P> const>
get_dim(std::shared_ptr<Layout<N> const> const & layout) { return layout; }


} // namespace detail
} // ndarray

#endif // !NDARRAY_detail_Layout_hpp_INCLUDED
