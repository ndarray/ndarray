// -*- c++ -*-
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
