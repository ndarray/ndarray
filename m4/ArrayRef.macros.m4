define(`GENERAL_ASSIGN',
`
    /// \brief $1 assignment of arrays and array expressions.
    template <typename Other>
    ArrayRef const &
    operator $1(ExpressionBase<Other> const & expr) const {
        LSST_NDARRAY_ASSERT(expr.getShape() 
                         == this->getShape().template first<ExpressionBase<Other>::ND::value>());
        indir(`$3',$1)
        return *this;
    }

    /// \brief $1 assignment of scalars.
    template <typename Scalar>
#ifndef DOXYGEN
    typename boost::enable_if<boost::is_convertible<Scalar,T>, ArrayRef const &>::type
#else
    ArrayRef const &
#endif
    operator $1(Scalar const & scalar) const {
        indir(`$2',$1)
        return *this;
    }')dnl
define(`BASIC_ASSIGN_SCALAR',`std::fill(this->begin(),this->end(),scalar);')dnl
define(`BASIC_ASSIGN_EXPR',`std::copy(expr.begin(),expr.end(),this->begin());')dnl
define(`AUGMENTED_ASSIGN_SCALAR',
`Iterator const i_end = this->end();
        for (Iterator i = this->begin(); i != i_end; ++i) (*i) $1 scalar;')dnl
define(`AUGMENTED_ASSIGN_EXPR',
`Iterator const i_end = this->end();
        typename Other::Iterator j = expr.begin();
        for (Iterator i = this->begin(); i != i_end; ++i, ++j) (*i) $1 (*j);')dnl
define(`BASIC_ASSIGN',`GENERAL_ASSIGN(`=',`BASIC_ASSIGN_SCALAR',`BASIC_ASSIGN_EXPR')')dnl
define(`AUGMENTED_ASSIGN',`GENERAL_ASSIGN($1,`AUGMENTED_ASSIGN_SCALAR',`AUGMENTED_ASSIGN_EXPR')')dnl
