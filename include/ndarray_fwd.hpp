#ifndef NDARRAY_ndarray_fwd_hpp_INCLUDED
#define NDARRAY_ndarray_fwd_hpp_INCLUDED

/**
 * @file ndarray_fwd.hpp 
 *
 * @brief Forward declarations and default template parameters for ndarray.
 */

/// \defgroup MainGroup Main

/// \defgroup OpGroup Operators

/// \defgroup VectorGroup Vectors

/// @internal \defgroup InternalGroup Internals

#include <boost/type_traits/is_const.hpp>
#include <boost/type_traits/add_const.hpp>
#include <boost/type_traits/remove_const.hpp>
#include <cassert>

#define NDARRAY_ASSERT(ARG) assert(ARG)

/** @namespace ndarray @brief Main public namespace */
namespace ndarray {

template <typename T, int N, int C> struct ArrayTraits;
template <typename Expression_> struct ExpressionTraits;

/** @internal @namespace ndarray::detail @brief Internal namespace */
namespace detail {

template <typename T, int N> class Core;

template <int N, typename Allocator_> struct AllocationInitializer;
template <typename T, int N> struct ExternalInitializer;

class CountingExpression;

template <
    typename Operand,
    typename UnaryFunction,
    int N = ExpressionTraits<Operand>::ND::value
    >
class UnaryOpExpression;

template <
    typename Operand1,
    typename Operand2,
    typename BinaryFunction,
    int N = ExpressionTraits<Operand1>::ND::value
    >
class BinaryOpExpression;

template <typename Iterator_> struct IteratorTraits;

template <typename T, int N, int C> class NestedIterator;

template <typename T> class StridedIterator;

template <
    typename Operand, 
    typename UnaryFunction
    >
class UnaryOpIterator;

template <
    typename Operand1,
    typename Operand2,
    typename BinaryFunction
    >
class BinaryOpIterator;

} // namespace ndarray::detail

template <typename Derived> class ExpressionBase;
template <typename Derived> class ArrayBase;
template <typename T, int N, int C=0> class ArrayRef;
template <typename T, int N, int C=0> class Array;
template <typename T, int N> class Vector;

} // namespace ndarray

#endif // !NDARRAY_ndarray_fwd_hpp_INCLUDED
