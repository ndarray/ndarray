// -*- lsst-c++ -*-
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
changecom(`###')dnl
define(`SPECIALIZE',
`
/**
 *  @brief An intermediate CRTP base class for Array and ArrayRef (specialization for $1).
 */
template <typename Derived>
class ArrayBaseN<Derived,$1> : public ArrayBase< Derived > {
    typedef ArrayBase<Derived> Super;
protected:
    typedef typename Super::Core Core;
    typedef typename Super::CorePtr CorePtr;
public:
    typedef typename Super::Element Element;

    Element & operator()($2) const {
        return this->operator[](makeVector($3));
    }

private:
    template <typename T_, int N_, int C_> friend class Array;
    template <typename T_, int N_, int C_> friend class ArrayRef;
    void operator=(ArrayBaseN const & other) {
        Super::operator=(other);
    }
    /// @internal @brief Construct an ArrayRef from a pointer and Core.
    ArrayBaseN(Element * data, CorePtr const & core) : Super(data, core) {}
};')dnl
#ifndef NDARRAY_ArrayBaseN_h_INCLUDED
#define NDARRAY_ArrayBaseN_h_INCLUDED

/** 
 *  @file ndarray/ArrayBaseN.h
 *
 *  @brief Definition of ArrayBaseN, a dimension-specialized CRTP base class for Array and ArrayRef.
 */

#include "ndarray/ArrayBase.h"

namespace ndarray {

/**
 *  @brief An intermediate CRTP base class for Array and ArrayRef.
 */
template <typename Derived, int N = ArrayBase<Derived>::ND::value>
class ArrayBaseN : public ArrayBase< Derived > {
    typedef ArrayBase<Derived> Super;
protected:
    typedef typename Super::Core Core;
    typedef typename Super::CorePtr CorePtr;
public:
    typedef typename Super::Element Element;
private:
    template <typename T_, int N_, int C_> friend class Array;
    template <typename T_, int N_, int C_> friend class ArrayRef;
    void operator=(ArrayBaseN const & other) {
        Super::operator=(other);
    }
    /// @internal @brief Construct an ArrayRef from a pointer and Core.
    ArrayBaseN(Element * data, CorePtr const & core) : Super(data, core) {}
};

SPECIALIZE(1, `int n0', `n0')
SPECIALIZE(2, `int n0, int n1', `n0, n1')
SPECIALIZE(3, `int n0, int n1, int n2', `n0, n1, n2')
SPECIALIZE(4, `int n0, int n1, int n2, int n3', `n0, n1, n2, n3')
SPECIALIZE(5, `int n0, int n1, int n2, int n3, int n4', `n0, n1, n2, n3, n4')
SPECIALIZE(6, `int n0, int n1, int n2, int n3, int n4, int n5', `n0, n1, n2, n3, n4, n5')

} // namespace ndarray

#endif // !NDARRAY_ArrayBaseN_h_INCLUDED
