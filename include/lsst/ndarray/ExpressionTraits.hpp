#ifndef LSST_NDARRAY_ExpressionTraits_hpp_INCLUDED
#define LSST_NDARRAY_ExpressionTraits_hpp_INCLUDED

/** 
 *  @file lsst/ndarray/ExpressionTraits.hpp
 *
 *  @brief Traits for Expression.
 */

#include "lsst/ndarray_fwd.hpp"
#include <boost/static_assert.hpp>

namespace lsst { namespace ndarray {

/**
 *  @brief Traits for expressions.
 *
 *  @ingroup MainGroup
 */
template <typename Expression_> struct ExpressionTraits {
    typedef boost::mpl::true_ IsScalar;
};

/**
 *  @internal @brief ExpressionTraits specialization for UnaryOpExpression.
 *
 *  @ingroup InternalGroup
 */
template <typename Operand, typename UnaryFunction, int N>
struct ExpressionTraits< detail::UnaryOpExpression<Operand,UnaryFunction,N> > {
    typedef typename UnaryFunction::result_type Element;
    typedef typename ExpressionTraits<Operand>::ND ND;
    typedef detail::UnaryOpIterator<Operand,UnaryFunction> Iterator;
    typedef detail::UnaryOpExpression<
        typename ExpressionTraits<Operand>::Reference,UnaryFunction,N-1
        > Value;
    typedef Value Reference;
    typedef boost::mpl::false_ IsScalar;
};

/**
 *  @internal @brief ExpressionTraits specialization for 1D UnaryOpExpression.
 *
 *  @ingroup InternalGroup
 */
template <typename Operand, typename UnaryFunction>
struct ExpressionTraits< detail::UnaryOpExpression<Operand,UnaryFunction,1> > {
    typedef typename UnaryFunction::result_type Element;
    typedef typename ExpressionTraits<Operand>::ND ND;
    typedef detail::UnaryOpIterator<Operand,UnaryFunction> Iterator;
    typedef Element Reference;
    typedef Element Value;
    typedef boost::mpl::false_ IsScalar;
};

/**
 *  @internal @brief ExpressionTraits specialization for BinaryOpExpression.
 *
 *  @ingroup InternalGroup
 */
template <typename Operand1, typename Operand2, typename BinaryFunction, int N>
struct ExpressionTraits< detail::BinaryOpExpression<Operand1,Operand2,BinaryFunction,N> > {
    typedef typename BinaryFunction::result_type Element;
    typedef typename ExpressionTraits<Operand1>::ND ND;
    typedef detail::BinaryOpIterator<Operand1,Operand2,BinaryFunction> Iterator;
    typedef detail::BinaryOpExpression<
        typename ExpressionTraits<Operand1>::Reference,
        typename ExpressionTraits<Operand2>::Reference,
        BinaryFunction, N-1 > Reference;
    typedef Reference Value;
    typedef boost::mpl::false_ IsScalar;
    BOOST_STATIC_ASSERT((ND::value == ExpressionTraits<Operand2>::ND::value));
};

/**
 *  @internal @brief ExpressionTraits specialization for 1D BinaryOpExpression.
 *
 *  @ingroup InternalGroup
 */
template <typename Operand1, typename Operand2, typename BinaryFunction>
struct ExpressionTraits< detail::BinaryOpExpression<Operand1,Operand2,BinaryFunction,1> > {
    typedef typename BinaryFunction::result_type Element;
    typedef typename ExpressionTraits<Operand1>::ND ND;
    typedef detail::BinaryOpIterator<Operand1,Operand2,BinaryFunction> Iterator;
    typedef Element Value;
    typedef Element Reference;
    typedef boost::mpl::false_ IsScalar;
    BOOST_STATIC_ASSERT((ND::value == ExpressionTraits<Operand2>::ND::value));
};

}} // namespace lsst::ndarray

#endif // !LSST_NDARRAY_ExpressionTraits_hpp_INCLUDED
