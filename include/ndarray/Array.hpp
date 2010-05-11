#ifndef NDARRAY_Array_hpp_INCLUDED
#define NDARRAY_Array_hpp_INCLUDED

/** 
 *  @file ndarray/Array.hpp
 *
 *  @brief Definitions for Array.
 */

#include "ndarray_fwd.hpp"
#include "ndarray/ArrayTraits.hpp"
#include "ndarray/ArrayBase.hpp"
#include "ndarray/Vector.hpp"
#include "ndarray/detail/Core.hpp"
#include "ndarray/views.hpp"

namespace ndarray {

/**
 *  @brief A multidimensional strided array.
 *
 *  Array is the workhorse class of the ndarray library.
 */
template <typename T, int N, int C>
class Array : public ArrayBase< Array<T,N,C> > {
    typedef ArrayBase<Array> Super;
    typedef typename Super::Core Core;
    typedef typename Super::CorePtr CorePtr;
public:

    /** 
     *  @brief Default constructor. 
     *
     *  Creates an empty array with zero dimensions and null memory.
     */
    Array() : Super(0, Core::create()) {}

    /**
     *  @brief Non-converting copy constructor.
     */
    Array(Array const & other) : Super(other._data, other._core) {}

    /**
     *  @brief Converting copy constructor. 
     *
     *  Implicit conversion is allowed for non-const to const and for 
     *  more guaranteed RMC to less guaranteed RMC (see \ref overview).
     */
    template <typename T_, int C_>
    Array(
        Array<T_,N,C_> const & other
#ifndef DOXYGEN
        , typename boost::enable_if_c<((C_>=C) && boost::is_convertible<T_*,T*>::value),void*>::type=0
#endif
    ) : Super(other._data, other._core) {}

    /**
     *  @brief Converting copy constructor. 
     *
     *  Implicit conversion is allowed for non-const to const and for 
     *  more guaranteed RMC to less guaranteed RMC (see \ref overview).
     */
    template <typename T_, int C_>
    Array(
        ArrayRef<T_,N,C_> const & other
#ifndef DOXYGEN
        , typename boost::enable_if_c<((C_>=C) && boost::is_convertible<T_*,T*>::value),void*>::type=0
#endif
    ) : Super(other._data, other._core) {}

    /**
     *  @brief Non-converting shallow assignment.
     */
    Array & operator=(Array const & other) {
        if (&other != this) {
            this->_data = other._data;
            this->_core = other._core;
        }
        return *this;
    }

    /**
     *  @brief Converting shallow assignment. 
     *
     *  Implicit conversion is allowed for non-const -> const and for 
     *  more guaranteed RMC -> less guaranteed RMC (see \ref overview).
     */
    template <typename T_, int C_>
#ifndef DOXYGEN
    typename boost::enable_if_c<((C_>=C) && boost::is_convertible<T_*,T*>::value), Array &>::type
#else
    Array &
#endif
    operator=(Array<T_,N,C_> const & other) {
        this->_data = other._data;
        this->_core = other._core;
        return *this;
    }

    /**
     *  @brief Converting shallow assignment. 
     *
     *  Implicit conversion is allowed for non-const -> const and for 
     *  more guaranteed RMC -> less guaranteed RMC (see \ref overview).
     */
    template <typename T_, int C_>
#ifndef DOXYGEN
    typename boost::enable_if_c<((C_>=C) && boost::is_convertible<T_*,T*>::value), Array &>::type
#else
    Array &
#endif
    operator=(ArrayRef<T_,N,C_> const & other) {
        this->_data = other._data;
        this->_core = other._core;
        return *this;
    }

    /**
     *  @brief Shallow equality comparison: return true if the arrays share data and
     *         have the same shape and strides. 
     */
    template <typename T_, int N_, int C_>
    bool operator==(Array<T_,N_,C_> const & other) const {
        return this->getData() == other.getData()
            && this->getShape() == other.getShape()
            && this->getStrides() == other.getStrides();
    }

    /**
     *  @brief Shallow inequality comparison. 
     */
    template <typename T_, int N_, int C_>
    bool operator!=(Array<T_,N_,C_> const & other) const {
        return !this->operator==(other);
    }

private:
    template <typename T_, int N_, int C_> friend class Array;
    template <typename T_, int N_, int C_> friend class ArrayRef;
    template <typename T_, int N_, int C_> friend struct ArrayTraits;
    template <typename Derived> friend class ArrayBase;
    template <typename Array_> friend struct detail::ArrayAccess;

    /// @internal @brief Construct an Array from a pointer and Core.
    Array(T * data, CorePtr const & core) : Super(data, core) {}
};

} // namespace ndarray

#endif // !NDARRAY_Array_hpp_INCLUDED
