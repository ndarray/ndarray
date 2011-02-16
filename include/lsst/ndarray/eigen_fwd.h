/* 
 * LSST Data Management System
 * Copyright 2008, 2009, 2010 LSST Corporation.
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

#ifndef LSST_NDARRAY_eigen_fwd_hpp_INCLUDED
#define LSST_NDARRAY_eigen_fwd_hpp_INCLUDED

/**
 *  @file lsst/ndarray/eigen_fwd.hpp
 *  @brief Forward declarations for ndarray/eigen interface.
 *
 *  \note This file is not included by the main "lsst/ndarray.h" header file.
 */

/** 
 * \defgroup EigenGroup Eigen
 * Interoperability with the Eigen 2 linear algebra library.
 */

#include "lsst/ndarray_fwd.h"
#include <Eigen/Core>

namespace lsst { namespace ndarray {

template <typename T, int N, int C>
class EigenView;

template <typename T, int N, int C>
class TransposedEigenView;

}} // namespace lsst::ndarray

#endif // !LSST_NDARRAY_eigen_fwd_hpp_INCLUDED
