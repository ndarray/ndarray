#ifndef NDARRAY_eigen_fwd_hpp_INCLUDED
#define NDARRAY_eigen_fwd_hpp_INCLUDED

/**
 *  @file ndarray/eigen_fwd.hpp
 *  @brief Forward declarations for ndarray/eigen interface.
 *
 *  \note This file is not included by the main "ndarray.hpp" header file.
 */

/** 
 * \defgroup EigenGroup Eigen
 * Interoperability with the Eigen 2 linear algebra library.
 */

#include "ndarray_fwd.hpp"
#include <Eigen/Core>

namespace ndarray {

template <typename T, int N, int C>
class EigenView;

template <typename T, int N, int C>
class TransposedEigenView;

} // namespace ndarray

#endif // !NDARRAY_eigen_fwd_hpp_INCLUDED
