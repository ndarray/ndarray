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
#ifndef LSST_NDARRAY_ndarray_fwd_h_INCLUDED
#define LSST_NDARRAY_ndarray_fwd_h_INCLUDED

/**
 * @file lsst/ndarray_fwd.h 
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

#ifdef __GNUC__
#if __GNUC__ == 4 && __GNUC_MINOR__ == 5
#define GCC_45
#endif
#endif

#define LSST_NDARRAY_ASSERT(ARG) assert(ARG)

/** @namespace ndarray @brief Main public namespace */
namespace lsst { namespace ndarray {

template <typename T, int N, int C> struct ArrayTraits;
template <typename Expression_> struct ExpressionTraits;
class Manager;

/** @internal @namespace lsst::ndarray::detail @brief Internal namespace */
namespace detail {

template <int N> class Core;

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

#ifndef GCC_45

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

#endif

} // namespace detail

template <typename Derived> class ExpressionBase;
template <typename Derived> class ArrayBase;
template <typename T, int N, int C=0> class ArrayRef;
template <typename T, int N, int C=0> class Array;
template <typename T, int N> struct Vector;

}} // namespace lsst::ndarray

#endif // !LSST_NDARRAY_ndarray_fwd_h_INCLUDED
