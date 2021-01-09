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
#ifndef NDARRAY_ArrayRef_h_INCLUDED
#define NDARRAY_ArrayRef_h_INCLUDED

/** 
 *  @file ndarray/ArrayRef.h
 *
 *  @brief Definitions for ArrayRef.
 */

#include "ndarray_fwd.h"
#include "ndarray/ArrayTraits.h"
#include "ndarray/ArrayBaseN.h"
#include "ndarray/detail/ArrayAccess.h"
#include "ndarray/Vector.h"
#include "ndarray/detail/Core.h"
#include "ndarray/views.h"

namespace ndarray {

/**
 *  @brief A proxy class for Array with deep assignment operators.
 */
template <typename T, int N, int C>
class ArrayRef : public ArrayBaseN< ArrayRef<T,N,C> > {
    typedef ArrayBaseN<ArrayRef> Super;
    typedef typename Super::Core Core;
    typedef typename Super::CorePtr CorePtr;
public:
    typedef typename Super::Iterator Iterator;

    /**
     *  @brief Construct an array with the given dimensions and allocated but uninitialized memory.
     *
     *  Unspecified dimensions will have unit size, and if the number of argmuments is greater
     *  than the number of dimensions of the array, the extra arguments will be silently ignored.
     *
     *  This is implemented in initialization.h.
     */
    explicit ArrayRef(Size n1, Size n2=1, Size n3=1, Size n4=1, Size n5=1, Size n6=1, Size n7=1, Size n8=1);

    /**
     *  @brief Construct an array with the given dimensions and allocated but uninitialized memory.
     *
     *  This is implemented in initialization.h.
     */
    template <typename U>
    explicit ArrayRef(Vector<U,N> const & shape);

    /**
     *  @brief Non-converting copy constructor.
     */
    ArrayRef(ArrayRef const & other) : Super(other._data, other._core) {}

    /**
     *  @brief Converting copy constructor. 
     *
     *  Implicit conversion is allowed for non-const to const and for 
     *  more guaranteed RMC to less guaranteed RMC (see \ref ndarrayTutorial).
     */
    template <typename T_, int C_>
    explicit ArrayRef(
        Array<T_,N,C_> const & other
#ifndef DOXYGEN
        , typename boost::enable_if<detail::Convertible<N,T_,C_,T,C>,void*>::type=0
#endif
    ) : Super(other._data, other._core) {}

    /**
     *  @brief Converting copy constructor. 
     *
     *  Implicit conversion is allowed for non-const to const and for 
     *  more guaranteed RMC to less guaranteed RMC (see \ref ndarrayTutorial).
     */
    template <typename T_, int C_>
    ArrayRef(
        ArrayRef<T_,N,C_> const & other
#ifndef DOXYGEN
        , typename boost::enable_if<detail::Convertible<N,T_,C_,T,C>,void*>::type=0
#endif
    ) : Super(other._data, other._core) {}

    /**
     *  @name Assignment and Augmented Assignment Operators
     *
     *  ArrayRef assignment is deep, and requires that
     *  the ArrayRef being assigned to has the same shape as
     *  the input array expression.  Scalar assignment sets
     *  all elements of the ArrayRef to a single value.
     */
    /// @{
    ArrayRef const & operator=(Array<T,N,C> const & other) const {
        NDARRAY_ASSERT(other.getShape() == this->getShape());
        std::copy(other.begin(), other.end(), this->begin());
        return *this;
    }

    ArrayRef const & operator=(ArrayRef const & other) const {
        NDARRAY_ASSERT(other.getShape() == this->getShape());
        std::copy(other.begin(), other.end(), this->begin());
        return *this;
    }


    /// \brief = assignment of arrays and array expressions.
    template <typename Other>
    ArrayRef const &
    operator =(ExpressionBase<Other> const & expr) const {
        NDARRAY_ASSERT(expr.getShape() 
                         == this->getShape().template first<ExpressionBase<Other>::ND::value>());
        std::copy(expr.begin(),expr.end(),this->begin());
        return *this;
    }

    /// \brief = assignment of scalars.
    template <typename Scalar>
#ifndef DOXYGEN
    typename boost::enable_if<boost::is_convertible<Scalar,T>, ArrayRef const &>::type
#else
    ArrayRef const &
#endif
    operator =(Scalar const & scalar) const {
        Super::Traits::fill(this->begin(),this->end(),scalar);
        return *this;
    }

    /// \brief += assignment of arrays and array expressions.
    template <typename Other>
    ArrayRef const &
    operator +=(ExpressionBase<Other> const & expr) const {
        NDARRAY_ASSERT(expr.getShape() 
                         == this->getShape().template first<ExpressionBase<Other>::ND::value>());
        Iterator const i_end = this->end();
        typename Other::Iterator j = expr.begin();
        for (Iterator i = this->begin(); i != i_end; ++i, ++j) (*i) += (*j);
        return *this;
    }

    /// \brief += assignment of scalars.
    template <typename Scalar>
#ifndef DOXYGEN
    typename boost::enable_if<boost::is_convertible<Scalar,T>, ArrayRef const &>::type
#else
    ArrayRef const &
#endif
    operator +=(Scalar const & scalar) const {
        Iterator const i_end = this->end();
        for (Iterator i = this->begin(); i != i_end; ++i) (*i) += scalar;
        return *this;
    }

    /// \brief -= assignment of arrays and array expressions.
    template <typename Other>
    ArrayRef const &
    operator -=(ExpressionBase<Other> const & expr) const {
        NDARRAY_ASSERT(expr.getShape() 
                         == this->getShape().template first<ExpressionBase<Other>::ND::value>());
        Iterator const i_end = this->end();
        typename Other::Iterator j = expr.begin();
        for (Iterator i = this->begin(); i != i_end; ++i, ++j) (*i) -= (*j);
        return *this;
    }

    /// \brief -= assignment of scalars.
    template <typename Scalar>
#ifndef DOXYGEN
    typename boost::enable_if<boost::is_convertible<Scalar,T>, ArrayRef const &>::type
#else
    ArrayRef const &
#endif
    operator -=(Scalar const & scalar) const {
        Iterator const i_end = this->end();
        for (Iterator i = this->begin(); i != i_end; ++i) (*i) -= scalar;
        return *this;
    }

    /// \brief *= assignment of arrays and array expressions.
    template <typename Other>
    ArrayRef const &
    operator *=(ExpressionBase<Other> const & expr) const {
        NDARRAY_ASSERT(expr.getShape() 
                         == this->getShape().template first<ExpressionBase<Other>::ND::value>());
        Iterator const i_end = this->end();
        typename Other::Iterator j = expr.begin();
        for (Iterator i = this->begin(); i != i_end; ++i, ++j) (*i) *= (*j);
        return *this;
    }

    /// \brief *= assignment of scalars.
    template <typename Scalar>
#ifndef DOXYGEN
    typename boost::enable_if<boost::is_convertible<Scalar,T>, ArrayRef const &>::type
#else
    ArrayRef const &
#endif
    operator *=(Scalar const & scalar) const {
        Iterator const i_end = this->end();
        for (Iterator i = this->begin(); i != i_end; ++i) (*i) *= scalar;
        return *this;
    }

    /// \brief /= assignment of arrays and array expressions.
    template <typename Other>
    ArrayRef const &
    operator /=(ExpressionBase<Other> const & expr) const {
        NDARRAY_ASSERT(expr.getShape() 
                         == this->getShape().template first<ExpressionBase<Other>::ND::value>());
        Iterator const i_end = this->end();
        typename Other::Iterator j = expr.begin();
        for (Iterator i = this->begin(); i != i_end; ++i, ++j) (*i) /= (*j);
        return *this;
    }

    /// \brief /= assignment of scalars.
    template <typename Scalar>
#ifndef DOXYGEN
    typename boost::enable_if<boost::is_convertible<Scalar,T>, ArrayRef const &>::type
#else
    ArrayRef const &
#endif
    operator /=(Scalar const & scalar) const {
        Iterator const i_end = this->end();
        for (Iterator i = this->begin(); i != i_end; ++i) (*i) /= scalar;
        return *this;
    }

    /// \brief %= assignment of arrays and array expressions.
    template <typename Other>
    ArrayRef const &
    operator %=(ExpressionBase<Other> const & expr) const {
        NDARRAY_ASSERT(expr.getShape() 
                         == this->getShape().template first<ExpressionBase<Other>::ND::value>());
        Iterator const i_end = this->end();
        typename Other::Iterator j = expr.begin();
        for (Iterator i = this->begin(); i != i_end; ++i, ++j) (*i) %= (*j);
        return *this;
    }

    /// \brief %= assignment of scalars.
    template <typename Scalar>
#ifndef DOXYGEN
    typename boost::enable_if<boost::is_convertible<Scalar,T>, ArrayRef const &>::type
#else
    ArrayRef const &
#endif
    operator %=(Scalar const & scalar) const {
        Iterator const i_end = this->end();
        for (Iterator i = this->begin(); i != i_end; ++i) (*i) %= scalar;
        return *this;
    }

    /// \brief ^= assignment of arrays and array expressions.
    template <typename Other>
    ArrayRef const &
    operator ^=(ExpressionBase<Other> const & expr) const {
        NDARRAY_ASSERT(expr.getShape() 
                         == this->getShape().template first<ExpressionBase<Other>::ND::value>());
        Iterator const i_end = this->end();
        typename Other::Iterator j = expr.begin();
        for (Iterator i = this->begin(); i != i_end; ++i, ++j) (*i) ^= (*j);
        return *this;
    }

    /// \brief ^= assignment of scalars.
    template <typename Scalar>
#ifndef DOXYGEN
    typename boost::enable_if<boost::is_convertible<Scalar,T>, ArrayRef const &>::type
#else
    ArrayRef const &
#endif
    operator ^=(Scalar const & scalar) const {
        Iterator const i_end = this->end();
        for (Iterator i = this->begin(); i != i_end; ++i) (*i) ^= scalar;
        return *this;
    }

    /// \brief &= assignment of arrays and array expressions.
    template <typename Other>
    ArrayRef const &
    operator &=(ExpressionBase<Other> const & expr) const {
        NDARRAY_ASSERT(expr.getShape() 
                         == this->getShape().template first<ExpressionBase<Other>::ND::value>());
        Iterator const i_end = this->end();
        typename Other::Iterator j = expr.begin();
        for (Iterator i = this->begin(); i != i_end; ++i, ++j) (*i) &= (*j);
        return *this;
    }

    /// \brief &= assignment of scalars.
    template <typename Scalar>
#ifndef DOXYGEN
    typename boost::enable_if<boost::is_convertible<Scalar,T>, ArrayRef const &>::type
#else
    ArrayRef const &
#endif
    operator &=(Scalar const & scalar) const {
        Iterator const i_end = this->end();
        for (Iterator i = this->begin(); i != i_end; ++i) (*i) &= scalar;
        return *this;
    }

    /// \brief |= assignment of arrays and array expressions.
    template <typename Other>
    ArrayRef const &
    operator |=(ExpressionBase<Other> const & expr) const {
        NDARRAY_ASSERT(expr.getShape() 
                         == this->getShape().template first<ExpressionBase<Other>::ND::value>());
        Iterator const i_end = this->end();
        typename Other::Iterator j = expr.begin();
        for (Iterator i = this->begin(); i != i_end; ++i, ++j) (*i) |= (*j);
        return *this;
    }

    /// \brief |= assignment of scalars.
    template <typename Scalar>
#ifndef DOXYGEN
    typename boost::enable_if<boost::is_convertible<Scalar,T>, ArrayRef const &>::type
#else
    ArrayRef const &
#endif
    operator |=(Scalar const & scalar) const {
        Iterator const i_end = this->end();
        for (Iterator i = this->begin(); i != i_end; ++i) (*i) |= scalar;
        return *this;
    }

    /// \brief <<= assignment of arrays and array expressions.
    template <typename Other>
    ArrayRef const &
    operator <<=(ExpressionBase<Other> const & expr) const {
        NDARRAY_ASSERT(expr.getShape() 
                         == this->getShape().template first<ExpressionBase<Other>::ND::value>());
        Iterator const i_end = this->end();
        typename Other::Iterator j = expr.begin();
        for (Iterator i = this->begin(); i != i_end; ++i, ++j) (*i) <<= (*j);
        return *this;
    }

    /// \brief <<= assignment of scalars.
    template <typename Scalar>
#ifndef DOXYGEN
    typename boost::enable_if<boost::is_convertible<Scalar,T>, ArrayRef const &>::type
#else
    ArrayRef const &
#endif
    operator <<=(Scalar const & scalar) const {
        Iterator const i_end = this->end();
        for (Iterator i = this->begin(); i != i_end; ++i) (*i) <<= scalar;
        return *this;
    }

    /// \brief >>= assignment of arrays and array expressions.
    template <typename Other>
    ArrayRef const &
    operator >>=(ExpressionBase<Other> const & expr) const {
        NDARRAY_ASSERT(expr.getShape() 
                         == this->getShape().template first<ExpressionBase<Other>::ND::value>());
        Iterator const i_end = this->end();
        typename Other::Iterator j = expr.begin();
        for (Iterator i = this->begin(); i != i_end; ++i, ++j) (*i) >>= (*j);
        return *this;
    }

    /// \brief >>= assignment of scalars.
    template <typename Scalar>
#ifndef DOXYGEN
    typename boost::enable_if<boost::is_convertible<Scalar,T>, ArrayRef const &>::type
#else
    ArrayRef const &
#endif
    operator >>=(Scalar const & scalar) const {
        Iterator const i_end = this->end();
        for (Iterator i = this->begin(); i != i_end; ++i) (*i) >>= scalar;
        return *this;
    }
    ///@}

private:
    template <typename T_, int N_, int C_> friend class Array;
    template <typename T_, int N_, int C_> friend class ArrayRef;
    template <typename T_, int N_, int C_> friend struct ArrayTraits;
    template <typename Derived> friend class ArrayBase;
    template <typename Array_> friend class detail::ArrayAccess;

    /// @internal @brief Construct an ArrayRef from a pointer and Core.
    ArrayRef(T * data, CorePtr const & core) : Super(data, core) {}

};

} // namespace ndarray

#endif // !NDARRAY_ArrayRef_h_INCLUDED
