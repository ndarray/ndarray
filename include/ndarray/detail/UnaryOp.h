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
#ifndef NDARRAY_DETAIL_UnaryOp_h_INCLUDED
#define NDARRAY_DETAIL_UnaryOp_h_INCLUDED

/** 
 *  @file ndarray/detail/UnaryOp.h
 *
 *  @brief Lazy unary expression templates.
 */

#include "ndarray/ExpressionBase.h"
#include "ndarray/vectorize.h"
#include <boost/iterator/iterator_adaptor.hpp>

namespace ndarray {
namespace detail {

/**
 *  @internal @brief An iterator for unary expression templates.
 *
 *  @ingroup ndarrayInternalGroup
 *
 *  Acts as a "transform" iterator.
 */
template <typename Operand, typename UnaryFunction>
class UnaryOpIterator : public boost::iterator_adaptor<
    UnaryOpIterator<Operand,UnaryFunction>,
    typename ExpressionTraits<Operand>::Iterator,
    typename ExpressionTraits< UnaryOpExpression<Operand,UnaryFunction> >::Value,
    boost::use_default,
    typename ExpressionTraits< UnaryOpExpression<Operand,UnaryFunction> >::Reference
    > {
    typedef UnaryOpExpression<Operand,UnaryFunction> Operation;
public:
    typedef typename ExpressionTraits<Operand>::Iterator BaseIterator;
    typedef typename ExpressionTraits<Operation>::Value Value;
    typedef typename ExpressionTraits<Operation>::Reference Reference;

    UnaryOpIterator() : UnaryOpIterator::iterator_adaptor_(), _functor() {}

    UnaryOpIterator(BaseIterator const & baseIter, UnaryFunction const & functor) :
        UnaryOpIterator::iterator_adaptor_(baseIter), _functor(functor) {}

    UnaryOpIterator(UnaryOpIterator const & other) :
        UnaryOpIterator::iterator_adaptor_(other), _functor(other._functor) {}

private:
    friend class boost::iterator_core_access;

    Reference dereference() const {
        return vectorize(_functor,*this->base_reference());
    }
    
    UnaryFunction _functor;
};

/**
 *  @internal @brief A unary expression template.
 *
 *  @ingroup ndarrayInternalGroup
 *
 *  Represents the lazy evaluation of a unary expression.
 */
template <typename Operand, typename UnaryFunction, int N>
class UnaryOpExpression : public ExpressionBase< UnaryOpExpression<Operand,UnaryFunction,N> > {
    typedef UnaryOpExpression<Operand,UnaryFunction,N> Self;
public:
    typedef typename ExpressionTraits<Self>::Element Element;
    typedef typename ExpressionTraits<Self>::ND ND;
    typedef typename ExpressionTraits<Self>::Iterator Iterator;
    typedef typename ExpressionTraits<Self>::Value Value;
    typedef typename ExpressionTraits<Self>::Reference Reference;
    typedef Vector<int,N> Index;
    
    UnaryOpExpression(Operand const & operand, UnaryFunction const & functor) :
        _operand(operand), _functor(functor) {}

    Reference operator[](int n) const {
        return Reference(_operand[n],_functor);
    }

    Iterator begin() const {
        return Iterator(_operand.begin(),_functor);
    }

    Iterator end() const {
        return Iterator(_operand.end(),_functor);
    }

    template <int P> int getSize() const {
        return _operand.template getSize<P>();
    }

    Index getShape() const {
        return _operand.getShape();
    }

    Operand _operand;
    UnaryFunction _functor;
};

} // namespace detail
} // namespace ndarray

#endif // !NDARRAY_DETAIL_UnaryOp_h_INCLUDED
