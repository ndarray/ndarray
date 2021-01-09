// -*- c++ -*-
/*
 * Copyright (c) 2010-2012, Jim Bosch
 * All rights reserved.
 *
 * ndarray is distributed under a simple BSD-like license;
 * see the LICENSE file that should be present in the root
 * of the source distribution, or alternately available at:
 * https://github.com/ndarray/ndarray
 */

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


/**
 *  @brief An intermediate CRTP base class for Array and ArrayRef (specialization for 1).
 */
template <typename Derived>
class ArrayBaseN<Derived,1> : public ArrayBase< Derived > {
    typedef ArrayBase<Derived> Super;
protected:
    typedef typename Super::Core Core;
    typedef typename Super::CorePtr CorePtr;
public:
    typedef typename Super::Element Element;

    Element & operator()(int n0) const {
        return this->operator[](makeVector(n0));
    }

private:
    template <typename T_, int N_, int C_> friend class Array;
    template <typename T_, int N_, int C_> friend class ArrayRef;

    void operator=(ArrayBaseN const & other) {
        Super::operator=(other);
    }

    template <typename Other>
    ArrayBaseN(ArrayBaseN<Other,1> const & other) : Super(other) {}

    ArrayBaseN(Element * data, CorePtr const & core) : Super(data, core) {}
};

/**
 *  @brief An intermediate CRTP base class for Array and ArrayRef (specialization for 2).
 */
template <typename Derived>
class ArrayBaseN<Derived,2> : public ArrayBase< Derived > {
    typedef ArrayBase<Derived> Super;
protected:
    typedef typename Super::Core Core;
    typedef typename Super::CorePtr CorePtr;
public:
    typedef typename Super::Element Element;

    Element & operator()(int n0, int n1) const {
        return this->operator[](makeVector(n0, n1));
    }

private:
    template <typename T_, int N_, int C_> friend class Array;
    template <typename T_, int N_, int C_> friend class ArrayRef;

    void operator=(ArrayBaseN const & other) {
        Super::operator=(other);
    }

    template <typename Other>
    ArrayBaseN(ArrayBaseN<Other,2> const & other) : Super(other) {}

    ArrayBaseN(Element * data, CorePtr const & core) : Super(data, core) {}
};

/**
 *  @brief An intermediate CRTP base class for Array and ArrayRef (specialization for 3).
 */
template <typename Derived>
class ArrayBaseN<Derived,3> : public ArrayBase< Derived > {
    typedef ArrayBase<Derived> Super;
protected:
    typedef typename Super::Core Core;
    typedef typename Super::CorePtr CorePtr;
public:
    typedef typename Super::Element Element;

    Element & operator()(int n0, int n1, int n2) const {
        return this->operator[](makeVector(n0, n1, n2));
    }

private:
    template <typename T_, int N_, int C_> friend class Array;
    template <typename T_, int N_, int C_> friend class ArrayRef;

    void operator=(ArrayBaseN const & other) {
        Super::operator=(other);
    }

    template <typename Other>
    ArrayBaseN(ArrayBaseN<Other,3> const & other) : Super(other) {}

    ArrayBaseN(Element * data, CorePtr const & core) : Super(data, core) {}
};

/**
 *  @brief An intermediate CRTP base class for Array and ArrayRef (specialization for 4).
 */
template <typename Derived>
class ArrayBaseN<Derived,4> : public ArrayBase< Derived > {
    typedef ArrayBase<Derived> Super;
protected:
    typedef typename Super::Core Core;
    typedef typename Super::CorePtr CorePtr;
public:
    typedef typename Super::Element Element;

    Element & operator()(int n0, int n1, int n2, int n3) const {
        return this->operator[](makeVector(n0, n1, n2, n3));
    }

private:
    template <typename T_, int N_, int C_> friend class Array;
    template <typename T_, int N_, int C_> friend class ArrayRef;

    void operator=(ArrayBaseN const & other) {
        Super::operator=(other);
    }

    template <typename Other>
    ArrayBaseN(ArrayBaseN<Other,4> const & other) : Super(other) {}

    ArrayBaseN(Element * data, CorePtr const & core) : Super(data, core) {}
};

/**
 *  @brief An intermediate CRTP base class for Array and ArrayRef (specialization for 5).
 */
template <typename Derived>
class ArrayBaseN<Derived,5> : public ArrayBase< Derived > {
    typedef ArrayBase<Derived> Super;
protected:
    typedef typename Super::Core Core;
    typedef typename Super::CorePtr CorePtr;
public:
    typedef typename Super::Element Element;

    Element & operator()(int n0, int n1, int n2, int n3, int n4) const {
        return this->operator[](makeVector(n0, n1, n2, n3, n4));
    }

private:
    template <typename T_, int N_, int C_> friend class Array;
    template <typename T_, int N_, int C_> friend class ArrayRef;

    void operator=(ArrayBaseN const & other) {
        Super::operator=(other);
    }

    template <typename Other>
    ArrayBaseN(ArrayBaseN<Other,5> const & other) : Super(other) {}

    ArrayBaseN(Element * data, CorePtr const & core) : Super(data, core) {}
};

/**
 *  @brief An intermediate CRTP base class for Array and ArrayRef (specialization for 6).
 */
template <typename Derived>
class ArrayBaseN<Derived,6> : public ArrayBase< Derived > {
    typedef ArrayBase<Derived> Super;
protected:
    typedef typename Super::Core Core;
    typedef typename Super::CorePtr CorePtr;
public:
    typedef typename Super::Element Element;

    Element & operator()(int n0, int n1, int n2, int n3, int n4, int n5) const {
        return this->operator[](makeVector(n0, n1, n2, n3, n4, n5));
    }

private:
    template <typename T_, int N_, int C_> friend class Array;
    template <typename T_, int N_, int C_> friend class ArrayRef;

    void operator=(ArrayBaseN const & other) {
        Super::operator=(other);
    }

    template <typename Other>
    ArrayBaseN(ArrayBaseN<Other,6> const & other) : Super(other) {}

    ArrayBaseN(Element * data, CorePtr const & core) : Super(data, core) {}
};

} // namespace ndarray

#endif // !NDARRAY_ArrayBaseN_h_INCLUDED
