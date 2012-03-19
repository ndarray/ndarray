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
define(`FUNCTION_TAG',
`
    struct $1 : public AdaptableFunctionTag<$1> {
        template <typename A, typename B>
        struct ScalarFunction : public $2<A,B> {
            typename $2<A,B>::result_type operator()(
                typename $2<A,B>::ParamA a,
                typename $2<A,B>::ParamB b
            ) const {
                return a $3 b;
            }
        };
    };')dnl
define(`BINARY_OP',
`
    template <typename Operand, typename Scalar>
#ifndef DOXYGEN
    typename boost::enable_if<
        typename ExpressionTraits<Scalar>::IsScalar,
        detail::UnaryOpExpression< Operand, typename $1::template ExprScalar<Operand,Scalar>::Bound >
    >::type
#else
    <unspecified-expression-type>
#endif
    $2(ExpressionBase<Operand> const & operand, Scalar const & scalar) {
        return vectorize($1::template ExprScalar<Operand,Scalar>::bind(scalar),operand);
    }

    template <typename Operand, typename Scalar>
#ifndef DOXYGEN
    typename boost::enable_if<
        typename ExpressionTraits<Scalar>::IsScalar,
        detail::UnaryOpExpression< Operand, typename $1::template ScalarExpr<Operand,Scalar>::Bound >
    >::type
#else
    <unspecified-expression-type>
#endif
    $2(Scalar const & scalar, ExpressionBase<Operand> const & operand) {
        return vectorize($1::template ScalarExpr<Operand,Scalar>::bind(scalar),operand);
    }

    template <typename Operand1, typename Operand2>
#ifndef DOXYGEN
    detail::BinaryOpExpression< 
         Operand1, Operand2,
         typename $1::template ExprExpr<Operand1,Operand2>::BinaryFunction
    >
#else
    <unspecified-expression-type>
#endif
    $2(ExpressionBase<Operand1> const & operand1, ExpressionBase<Operand2> const & operand2) {
        return vectorize(
            typename $1::template ExprExpr<Operand1,Operand2>::BinaryFunction(),
            operand1,
            operand2
        );
    }')dnl
define(`UNARY_OP',
`
    template <typename Operand>
#ifndef DOXYGEN
    detail::UnaryOpExpression< Operand, $1<typename ExpressionTraits<Operand>::Element> >
#else
    <unspecified-expression-type>
#endif
    $2(ExpressionBase<Operand> const & operand) {
        return vectorize($1<typename ExpressionTraits<Operand>::Element>(),operand);
    }')dnl
#ifndef NDARRAY_operators_h_INCLUDED
#define NDARRAY_operators_h_INCLUDED

/** 
 *  \file ndarray/operators.h \brief Arithmetic and logical operators for Array.
 */

#include "ndarray/Array.h"
#include <boost/call_traits.hpp>
#include <boost/functional.hpp>

#include "ndarray/detail/UnaryOp.h"
#include "ndarray/detail/BinaryOp.h"
#include "ndarray/types.h"

namespace ndarray {

/// \cond INTERNAL
namespace detail {

/** 
 *  \internal @class PromotingBinaryFunction
 *  \brief A typedef-providing base class for binary functors with numeric type promotion.
 *
 *  \ingroup InternalGroup
 */
template <typename A, typename B>
struct PromotingBinaryFunction {
    typedef A ElementA;
    typedef B ElementB;
    typedef A first_argument_type;
    typedef B second_argument_type;
    typedef typename boost::call_traits<A>::param_type ParamA;
    typedef typename boost::call_traits<B>::param_type ParamB;
    typedef typename Promote<A,B>::Type result_type;
};

/** 
 *  \internal @class BinaryPredicate
 *  \brief A typedef-providing base class for binary predicates.
 *
 *  \ingroup InternalGroup
 */
template <typename A, typename B>
struct BinaryPredicate {
    typedef A ElementA;
    typedef B ElementB;
    typedef A first_argument_type;
    typedef B second_argument_type;
    typedef typename boost::call_traits<A>::param_type ParamA;
    typedef typename boost::call_traits<B>::param_type ParamB;
    typedef bool result_type;
};

/** 
 *  \internal @class AdaptableFunctionTag
 *  \brief A CRTP base class for non-template classes that contain a templated functor.
 *
 *  \ingroup InternalGroup
 */
template <typename Derived>
struct AdaptableFunctionTag {

    template <typename OperandB, typename A>
    struct ScalarExpr {
        typedef typename Derived::template ScalarFunction<
            A, typename ExpressionTraits<OperandB>::Element
            > BinaryFunction;
        typedef boost::binder1st<BinaryFunction> Bound;
        static Bound bind(A const & scalar) {
            return Bound(BinaryFunction(),scalar);
        }
    };

    template <typename OperandA, typename B>
    struct ExprScalar {
        typedef typename Derived::template ScalarFunction<
            typename ExpressionTraits<OperandA>::Element, B
            > BinaryFunction;
        typedef boost::binder2nd<BinaryFunction> Bound;
        static Bound bind(B const & scalar) {
            return Bound(BinaryFunction(),scalar);
        }
    };

    template <typename OperandA, typename OperandB>
    struct ExprExpr {
        typedef typename Derived::template ScalarFunction<
            typename ExpressionTraits<OperandA>::Element,
            typename ExpressionTraits<OperandB>::Element
            > BinaryFunction;
    };

};

/**
 *  \internal @class BitwiseNot
 *  \ingroup InternalGroup
 *  \brief An STL Unary Function class for bitwise NOT (unary ~).
 */
template <typename T>
struct BitwiseNot {
    typedef T argument_type;
    typedef T result_type;

    result_type operator()(argument_type arg) const { return ~arg; }
};

FUNCTION_TAG(PlusTag,PromotingBinaryFunction,+)
FUNCTION_TAG(MinusTag,PromotingBinaryFunction,-)
FUNCTION_TAG(MultipliesTag,PromotingBinaryFunction,*)
FUNCTION_TAG(DividesTag,PromotingBinaryFunction,/)
FUNCTION_TAG(ModulusTag,PromotingBinaryFunction,%)
FUNCTION_TAG(BitwiseXorTag,PromotingBinaryFunction,^)
FUNCTION_TAG(BitwiseOrTag,PromotingBinaryFunction,|)
FUNCTION_TAG(BitwiseAndTag,PromotingBinaryFunction,&)
FUNCTION_TAG(BitwiseLeftShiftTag,PromotingBinaryFunction,<<)
FUNCTION_TAG(BitwiseRightShiftTag,PromotingBinaryFunction,>>)

FUNCTION_TAG(EqualToTag,BinaryPredicate,==)
FUNCTION_TAG(NotEqualToTag,BinaryPredicate,!=)
FUNCTION_TAG(LessTag,BinaryPredicate,<)
FUNCTION_TAG(GreaterTag,BinaryPredicate,>)
FUNCTION_TAG(LessEqualTag,BinaryPredicate,<=)
FUNCTION_TAG(GreaterEqualTag,BinaryPredicate,>=)
FUNCTION_TAG(LogicalAnd,BinaryPredicate,&&)
FUNCTION_TAG(LogicalOr,BinaryPredicate,||)

} // namespace detail
/// \endcond

/// \addtogroup ndarrayOpGroup
/// @{
BINARY_OP(detail::PlusTag,operator+)
BINARY_OP(detail::MinusTag,operator-)
BINARY_OP(detail::MultipliesTag,operator*)
BINARY_OP(detail::DividesTag,operator/)
BINARY_OP(detail::ModulusTag,operator%)
BINARY_OP(detail::BitwiseXorTag,operator^)
BINARY_OP(detail::BitwiseOrTag,operator|)
BINARY_OP(detail::BitwiseAndTag,operator&)
BINARY_OP(detail::BitwiseLeftShiftTag,operator<<)
BINARY_OP(detail::BitwiseRightShiftTag,operator>>)

BINARY_OP(detail::EqualToTag, equal)
BINARY_OP(detail::NotEqualToTag, not_equal)
BINARY_OP(detail::LessTag, less)
BINARY_OP(detail::GreaterTag, greater)
BINARY_OP(detail::LessEqualTag, less_equal)
BINARY_OP(detail::GreaterEqualTag, great_equal)
BINARY_OP(detail::LogicalAnd, logical_and)
BINARY_OP(detail::LogicalOr, logical_or)

UNARY_OP(std::negate,operator-)
UNARY_OP(std::logical_not, logical_not)
UNARY_OP(detail::BitwiseNot,operator~)
/// @}

template <typename Scalar>
inline typename boost::enable_if<typename ExpressionTraits<Scalar>::IsScalar, bool>::type
any(Scalar const & scalar) {
    return bool(scalar);
}

/**
 *  \brief Return true if any of the elements of the given expression are true.
 *
 *  \ingroup MainGroup
 */
template <typename Derived>
inline bool
any(ExpressionBase<Derived> const & expr) {
    typename Derived::Iterator const i_end = expr.end();
    for (typename Derived::Iterator i = expr.begin(); i != i_end; ++i) {
        if (any(*i)) return true;
    }
    return false;
}

template <typename Scalar>
inline typename boost::enable_if<typename ExpressionTraits<Scalar>::IsScalar, bool>::type
all(Scalar const & scalar) {
    return bool(scalar);
}

/**
 *  \brief Return true if all of the elements of the given expression are true.
 *
 *  \ingroup MainGroup
 */
template <typename Derived>
inline bool
all(ExpressionBase<Derived> const & expr) {
    typename Derived::Iterator const i_end = expr.end();
    for (typename Derived::Iterator i = expr.begin(); i != i_end; ++i) {
        if (!all(*i)) return false;
    }
    return true;
}

template <typename Scalar1, typename Scalar2>
inline typename boost::enable_if<
    boost::mpl::and_<
        typename ExpressionTraits<Scalar1>::IsScalar,
        typename ExpressionTraits<Scalar2>::IsScalar
    >,
    bool
>::type
allclose(Scalar1 const & scalar1, Scalar2 const & scalar2, double tol=1E-8) {
    ApproximatelyEqual<Scalar1,Scalar2> func(tol);
    return func(scalar1, scalar2);
}

template <typename Scalar, typename Derived>
inline typename boost::enable_if<typename ExpressionTraits<Scalar>::IsScalar,bool>::type
allclose(Scalar const & scalar, ExpressionBase<Derived> const & expr, double tol=1E-8) {
    ApproximatelyEqual<Scalar,typename Derived::Element> func(tol);
    typename Derived::Iterator const i_end = expr.end();
    for (typename Derived::Iterator i = expr.begin(); i != i_end; ++i) {
        if (!allclose(scalar, *i, tol)) return false;
    }
    return true;
}

template <typename Scalar, typename Derived>
inline typename boost::enable_if<typename ExpressionTraits<Scalar>::IsScalar,bool>::type
allclose(ExpressionBase<Derived> const & expr, Scalar const & scalar, double tol=1E-8) {
    return allclose(scalar, expr, tol);
}

template <typename Derived1, typename Derived2>
inline bool
allclose(ExpressionBase<Derived1> const & expr1, ExpressionBase<Derived2> const & expr2, double tol=1E-8) {
    typename Derived1::Iterator const i_end = expr1.end();
    typename Derived1::Iterator i = expr1.begin();
    typename Derived2::Iterator j = expr2.begin();
    for (; i != i_end; ++i, ++j) {
        if (!allclose(*i, *j, tol)) return false;
    }
    return true;
}


template <typename Scalar>
inline typename boost::enable_if<typename ExpressionTraits<Scalar>::IsScalar, Scalar>::type
sum(Scalar const & scalar) { return scalar; }


/**
 *  \brief Return the sum of all elements of the given expression.
 *
 *  \ingroup MainGroup
 */
template <typename Derived>
inline typename Derived::Element
sum(ExpressionBase<Derived> const & expr) {
    typename Derived::Iterator const i_end = expr.end();
    typename Derived::Element total = static_cast<typename Derived::Element>(0);
    for (typename Derived::Iterator i = expr.begin(); i != i_end; ++i) {
        total += sum(*i);
    }
    return total;
}


} // namespace ndarray

#endif // !NDARRAY_operators_h_INCLUDED
