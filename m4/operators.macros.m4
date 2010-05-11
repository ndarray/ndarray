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
