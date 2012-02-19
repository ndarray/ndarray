// -*- c++ -*-
/*
 * Copyright 2012, Jim Bosch and the LSST Corporation
 * 
 * ndarray is available under two licenses, both of which are described
 * more fully in other files that should be distributed along with
 * the code:
 * 
 *  - A simple BSD-style license (ndarray-bsd-license.txt); under this
 *    license ndarray is broadly compatible with essentially any other
 *    code.
 * 
 *  - As a part of the LSST data management software system, ndarray is
 *    licensed with under the GPL v3 (LsstLicenseStatement.txt).
 * 
 * These files can also be found in the source distribution at:
 * 
 * https://github.com/ndarray/ndarray
 */
#ifndef NDARRAY_eigen_fwd_h_INCLUDED
#define NDARRAY_eigen_fwd_h_INCLUDED

/**
 *  @file ndarray/eigen_fwd.h
 *  @brief Forward declarations for ndarray/eigen interface.
 */

/** 
 * \defgroup ndarrayEigenGroup Eigen
 * Interoperability with the Eigen 3 linear algebra library.
 */

namespace Eigen {

struct MatrixXpr;

} // namespace Eigen

namespace ndarray {
namespace detail {

template <int N, int C, int Rows, int Cols> struct EigenStrideTraits;

} // namespace detail

template <
    typename T, int N, int C, 
    typename XprKind_=Eigen::MatrixXpr,
    int Rows_=-1,
    int Cols_=((N == 1) ? 1 : -1)
    >
class EigenView;

} // namespace ndarray

#endif // !NDARRAY_eigen_fwd_h_INCLUDED
