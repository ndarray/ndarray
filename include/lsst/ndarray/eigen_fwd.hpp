#ifndef LSST_NDARRAY_eigen_fwd_hpp_INCLUDED
#define LSST_NDARRAY_eigen_fwd_hpp_INCLUDED

/**
 *  @file lsst/ndarray/eigen_fwd.hpp
 *  @brief Forward declarations for ndarray/eigen interface.
 *
 *  \note This file is not included by the main "lsst/ndarray.hpp" header file.
 */

/** 
 * \defgroup EigenGroup Eigen
 * Interoperability with the Eigen 2 linear algebra library.
 */

#include "lsst/ndarray_fwd.hpp"
#include <Eigen/Core>

namespace lsst { namespace ndarray {

template <typename T, int N, int C>
class EigenView;

template <typename T, int N, int C>
class TransposedEigenView;

}} // namespace lsst::ndarray

#endif // !LSST_NDARRAY_eigen_fwd_hpp_INCLUDED
