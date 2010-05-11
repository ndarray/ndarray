#ifndef LSST_NDARRAY_DETAIL_UnaryOp_hpp_INCLUDED
#define LSST_NDARRAY_DETAIL_UnaryOp_hpp_INCLUDED

/** 
 *  @file lsst/ndarray/detail/UnaryOp.hpp
 *
 *  @brief Lazy unary expression templates.
 */

#include "lsst/ndarray/ExpressionBase.hpp"
#include "lsst/ndarray/vectorize.hpp"
#include <boost/iterator/iterator_adaptor.hpp>

namespace lsst { namespace ndarray {
namespace detail {

/**
 *  @internal @brief An iterator for unary expression templates.
 *
 *  @ingroup InternalGroup
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
 *  @ingroup InternalGroup
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

} // namespace lsst::ndarray::detail
}} // namespace lsst::ndarray

#endif // !LSST_NDARRAY_DETAIL_UnaryOp_hpp_INCLUDED
