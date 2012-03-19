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
#ifndef NDARRAY_DETAIL_BinaryOp_h_INCLUDED
#define NDARRAY_DETAIL_BinaryOp_h_INCLUDED

/** 
 *  @file ndarray/detail/BinaryOp.h
 *
 *  @brief Lazy binary expression templates.
 */

#include "ndarray/ExpressionBase.h"
#include "ndarray/vectorize.h"
#include <boost/iterator/iterator_adaptor.hpp>
#include <boost/iterator/zip_iterator.hpp>
#include <boost/tuple/tuple.hpp>

namespace ndarray {
namespace detail {

/**
 *  @internal @brief An iterator for binary expression templates.
 *
 *  @ingroup ndarrayInternalGroup
 *
 *  Acts as a combination "zip" and "transform" iterator.
 */
template <typename Operand1, typename Operand2, typename BinaryFunction>
class BinaryOpIterator : public boost::iterator_adaptor<
    BinaryOpIterator<Operand1,Operand2,BinaryFunction>,
    boost::zip_iterator<
        boost::tuple<
            typename ExpressionTraits<Operand1>::Iterator,
            typename ExpressionTraits<Operand2>::Iterator
            >
        >,
    typename ExpressionTraits< BinaryOpExpression<Operand1,Operand2,BinaryFunction> >::Value,
    boost::use_default,
    typename ExpressionTraits< BinaryOpExpression<Operand1,Operand2,BinaryFunction> >::Reference
    > {
    typedef BinaryOpExpression<Operand1,Operand2,BinaryFunction> Operation;
public:
    typedef typename ExpressionTraits<Operand1>::Iterator BaseIterator1;
    typedef typename ExpressionTraits<Operand2>::Iterator BaseIterator2;
    typedef typename ExpressionTraits<Operation>::Value Value;
    typedef typename ExpressionTraits<Operation>::Reference Reference;

    BinaryOpIterator() : BinaryOpIterator::iterator_adaptor_(), _functor() {}

    BinaryOpIterator(
        BaseIterator1 const & baseIter1, 
        BaseIterator2 const & baseIter2, 
        BinaryFunction const & functor
    ) :
        BinaryOpIterator::iterator_adaptor_(boost::make_tuple(baseIter1,baseIter2)),
        _functor(functor) {}

    BinaryOpIterator(BinaryOpIterator const & other) :
        BinaryOpIterator::iterator_adaptor_(other), _functor(other._functor) {}

private:
    friend class boost::iterator_core_access;

    Reference dereference() const {
        return vectorize(
            _functor, 
            this->base_reference()->template get<0>(),
            this->base_reference()->template get<1>()
        );
    }
    
    BinaryFunction _functor;
};

/**
 *  @internal @brief A binary expression template.
 *
 *  @ingroup ndarrayInternalGroup
 *
 *  Represents the lazy evaluation of a binary expression.
 */
template <typename Operand1, typename Operand2, typename BinaryFunction, int N>
class BinaryOpExpression : public ExpressionBase< BinaryOpExpression<Operand1,Operand2,BinaryFunction,N> > {
    typedef BinaryOpExpression<Operand1,Operand2,BinaryFunction,N> Self;
public:
    typedef typename ExpressionTraits<Self>::Element Element;
    typedef typename ExpressionTraits<Self>::ND ND;
    typedef typename ExpressionTraits<Self>::Iterator Iterator;
    typedef typename ExpressionTraits<Self>::Value Value;
    typedef typename ExpressionTraits<Self>::Reference Reference;
    typedef Vector<int,N> Index;
    
    BinaryOpExpression(
        Operand1 const & operand1,
        Operand2 const & operand2,
        BinaryFunction const & functor
    ) :
        _operand1(operand1), _operand2(operand2), _functor(functor) {
        NDARRAY_ASSERT(_operand1.getShape() == _operand2.getShape());
    }

    Reference operator[](int n) const {
        return Reference(_operand1[n],_operand2[n],_functor);
    }

    Iterator begin() const {
        return Iterator(_operand1.begin(),_operand2.begin(),_functor);
    }

    Iterator end() const {
        return Iterator(_operand1.end(),_operand2.end(),_functor);
    }

    template <int P> int getSize() const {
        return _operand1.template getSize<P>();
    }

    Index getShape() const {
        return _operand1.getShape();
    }

    Operand1 _operand1;
    Operand2 _operand2;
    BinaryFunction _functor;
};

} // namespace detail
} // namespace ndarray

#endif // !NDARRAY_DETAIL_BinaryOp_h_INCLUDED
