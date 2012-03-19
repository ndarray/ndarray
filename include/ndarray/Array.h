// -*- c++ -*-
/* 
 * LSST Data Management System
 * Copyright 2008, 2009, 2010, 2011 LSST Corporation.
 * 
 * This product includes software developed by the
 * LSST Project (http://www.lsst.org/).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the LSST License Statement and 
 * the GNU General Public License along with this program.  If not, 
 * see <http://www.lsstcorp.org/LegalNotices/>.
 */
#ifndef NDARRAY_Array_h_INCLUDED
#define NDARRAY_Array_h_INCLUDED

/** 
 *  @file ndarray/Array.h
 *
 *  @brief Definitions for Array.
 */

#include "ndarray_fwd.h"
#include "ndarray/ArrayTraits.h"
#include "ndarray/ArrayBaseN.h"
#include "ndarray/Vector.h"
#include "ndarray/detail/Core.h"
#include "ndarray/views.h"

namespace ndarray {

/**
 *  @brief A multidimensional strided array.
 *
 *  Array is the workhorse class of the ndarray library.
 */
template <typename T, int N, int C>
class Array : public ArrayBaseN< Array<T,N,C> > {
    typedef ArrayBaseN<Array> Super;
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
     *  more guaranteed RMC to less guaranteed RMC (see \ref index).
     */
    template <typename T_, int C_>
    Array(
        Array<T_,N,C_> const & other
#ifndef DOXYGEN
        , typename boost::enable_if<detail::Convertible<N,T_,C_,T,C>,void*>::type=0
#endif
    ) : Super(other._data, other._core) {}

    /**
     *  @brief Converting copy constructor. 
     *
     *  Implicit conversion is allowed for non-const to const and for 
     *  more guaranteed RMC to less guaranteed RMC (see \ref index).
     */
    template <typename T_, int C_>
    Array(
        ArrayRef<T_,N,C_> const & other
#ifndef DOXYGEN
        , typename boost::enable_if<detail::Convertible<N,T_,C_,T,C>,void*>::type=0
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
     *  more guaranteed RMC -> less guaranteed RMC (see \ref index).
     */
    template <typename T_, int C_>
#ifndef DOXYGEN
    typename boost::enable_if<detail::Convertible<N,T_,C_,T,C>, Array &>::type
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
     *  more guaranteed RMC -> less guaranteed RMC (see \ref index).
     */
    template <typename T_, int C_>
#ifndef DOXYGEN
    typename boost::enable_if<detail::Convertible<N,T_,C_,T,C>, Array &>::type
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

    /// @brief Lightweight shallow swap.
    void swap(Array & other) {
        std::swap(this->_data, other._data);
        this->_core.swap(other._core);
    }

    /**
     *  @brief Return true if the Array is definitely unique.
     *
     *  This will only return true if the manager overrides Manager::isUnique();
     *  this is true for the SimpleManager used by ndarray::allocate, but it is
     *  not true for ExternalManager.
     */
    bool isUnique() const { return this->_core->isUnique(); }

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

#endif // !NDARRAY_Array_h_INCLUDED
