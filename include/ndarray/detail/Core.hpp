#ifndef NDARRAY_DETAIL_Core_hpp_INCLUDED
#define NDARRAY_DETAIL_Core_hpp_INCLUDED

/**
 * @file ndarray/detail/Core.hpp 
 *
 * @brief Definitions for Core.
 */

#include <boost/intrusive_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/mpl/int.hpp>

#include "ndarray/Vector.hpp"

namespace ndarray {
namespace detail {

/**
 *  @internal
 *  @brief Internal data class for Array.
 *
 *  @ingroup InternalGroup
 *
 *  Core holds the shape, stride, and ownership data for an Array.
 *  A Core maintains its own reference count and can be shared
 *  by multiple Arrays via a const boost::intrusive pointer.
 *  
 *  Because a Core with N dimensions inherits from
 *  a Core with N-1 dimensions, subarrays can share a Core with
 *  their parants.
 *
 *  Core objects are never const; even an Array with a const
 *  template parameter holds a Core with a non-const template
 *  parameter.
 */
template <typename T, int N>
class Core : public Core<T,N-1> {
public:
    typedef T Element;  ///< data type
    typedef boost::mpl::int_<N> ND;  ///< number of dimensions
    typedef Core<T,N-1> Super;  ///< base class
    typedef boost::shared_ptr<Element> Owner;  ///< shared_ptr that owns the memory
    typedef boost::intrusive_ptr<Core> Ptr;            ///< non-const intrusive_ptr to Core
    typedef boost::intrusive_ptr<Core const> ConstPtr; ///< const intrusive_ptr to Core

    /// @brief Create a Core::Ptr with the given shape, strides, and owner.
    template <int M>
    static Ptr create(
        Vector<int,M> const & shape,
        Vector<int,M> const & strides, 
        Owner const & owner = Owner()
    ) {
        return Ptr(new Core(shape,strides,owner),false);
    }        

    /// @brief Create a Core::Ptr with the given shape and owner with RMC strides.
    template <int M>
    static Ptr create(
        Vector<int,M> const & shape,
        Owner const & owner = Owner()
    ) {
        return Ptr(new Core(shape,owner),false);
    }        

    /// @brief Create a Core::Ptr with the given owner and zero shape and strides.
    static Ptr create(
        Owner const & owner = Owner()
    ) {
        return Ptr(new Core(owner),false);
    }        

    /// @brief Return the size of the Nth dimension.
    int getSize() const { return _size; }

    /// @brief Return the stride of the Nth dimension.
    int getStride() const { return _stride; }

    /// @brief Set the size of the Nth dimension.
    void setSize(int size) { _size = size; }

    /// @brief Set the stride of the Nth dimension.
    void setStride(int stride) { _stride = stride; }

    /// @brief Recursively compute the offset to an element.
    template <int M>
    int computeOffset(Vector<int,M> const & index) const {
        return index[M-N] * this->getStride() + Super::computeOffset(index);
    }

    /// @brief Recursively fill a shape vector.
    template <int M>
    void fillShape(Vector<int,M> & shape) const {
        shape[M-N] = this->getSize();
        Super::fillShape(shape);
    }

    /// @brief Recursively fill a strides vector.
    template <int M>
    void fillStrides(Vector<int,M> & strides) const {
        strides[M-N] = this->getStride();
        Super::fillStrides(strides);
    }

    /// @brief Recursively determine the total number of elements.
    int getNumElements() const {
        return getSize() * Super::getNumElements();
    }
    
protected:

    template <int M>
    Core (
        Vector<int,M> const & shape,
        Vector<int,M> const & strides, 
        Owner const & owner
    ) : Super(shape,strides,owner), _size(shape[M-N]), _stride(strides[M-N]) {}

    template <int M>
    Core (
        Vector<int,M> const & shape,
        Owner const & owner
    ) : Super(shape,owner), _size(shape[M-N]), _stride(Super::getStride() * Super::getSize()) {}

    Core (
        Owner const & owner
    ) : Super(owner), _size(0), _stride(0) {}

private:
    int _size;
    int _stride;
};

/**
 *  @internal
 *  @brief Internal data class for Array, 0-D specialization.
 *
 *  @ingroup InternalGroup
 *
 *  The 0-D Core has size and stride == 1 and holds the reference
 *  count and owner; it is the base class for all other Cores.
 */
template <typename T>
class Core<T,0> {
public:
    typedef T Element;
    typedef boost::mpl::int_<0> ND;
    typedef boost::shared_ptr<Element> Owner;
    typedef boost::intrusive_ptr<Core> Ptr;
    typedef boost::intrusive_ptr<Core const> ConstPtr;

    friend inline void intrusive_ptr_add_ref(Core const * core) {
        ++core->_rc;
    }
 
    friend inline void intrusive_ptr_release(Core const * core) {
        if ((--core->_rc)==0) delete core;
    }

    int getSize() const { return 1; }
    int getStride() const { return 1; }

    /// @brief Recursively compute the offset to an element.
    template <int M>
    int computeOffset(Vector<int,M> const & index) const { return 0; }

    /// @brief Return the shared_ptr that manages the lifetime of the array data.
    Owner getOwner() const { return _owner; }

    /// @brief Set the shared_ptr that manages the lifetime of the array data.
    void setOwner(Owner const & owner) { _owner = owner; }

    /// @brief Recursively fill a shape vector.
    template <int M>
    void fillShape(Vector<int,M> const & shape) const {}

    /// @brief Recursively fill a strides vector.
    template <int M>
    void fillStrides(Vector<int,M> const & strides) const {}

    /// @brief Recursively determine the total number of elements.
    int getNumElements() const { return 1; }

    /// @brief Return the reference count (for debugging purposes).
    int getRC() const { return _rc; }

protected:

    virtual ~Core() {}

    template <int M>
    Core (
        Vector<int,M> const & shape,
        Vector<int,M> const & strides, 
        Owner const & owner
    ) : _owner(owner), _rc(1) {}

    template <int M>
    Core (
        Vector<int,M> const & shape,
        Owner const & owner
    ) : _owner(owner), _rc(1) {}

    Core (
        Owner const & owner
    ) : _owner(owner), _rc(1) {}

private:
    Owner _owner;
    mutable int _rc;
};


/**
 *  @internal @brief Cast a Core reference to a particular dimension.
 *
 *  @ingroup InternalGroup
 */
template <int P, typename T, int N>
inline Core<T,N-P> const & 
getDimension(Core<T,N> const & core) { return core; }

/**
 *  @internal @brief Cast a Core smart pointer to a particular dimension.
 *
 *  @ingroup InternalGroup
 */
template <int P, typename T, int N>
inline typename Core<T,N-P>::Ptr 
getDimension(typename Core<T,N>::Ptr const & core) { return core; }

} // namespace ndarray::detail
} // namespace ndarray

#endif // !NDARRAY_DETAIL_Core_hpp_INCLUDED
