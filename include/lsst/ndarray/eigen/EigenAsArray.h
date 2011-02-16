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
#ifndef LSST_NDARRAY_EIGEN_EigenAsArray_h_INCLUDED
#define LSST_NDARRAY_EIGEN_EigenAsArray_h_INCLUDED

/**
 *  @file lsst/ndarray/eigen/EigenAsArray.h
 *  @brief Functions that construct lsst::ndarray::Array views into Eigen objects.
 *
 *  \note This file is not included by the main "lsst/ndarray.h" header file.
 */

#include "lsst/ndarray/eigen_fwd.h"
#include "lsst/ndarray.h"
#include <Eigen/Core>

namespace lsst { namespace ndarray {

#ifndef DOXYGEN
template <typename Matrix>
typename boost::enable_if_c< 
    (Matrix::Flags & Eigen::DirectAccessBit) && (Matrix::Flags & Eigen::LinearAccessBit), 
    ArrayRef<typename boost::mpl::if_<boost::is_const<Matrix>,
                                      typename Matrix::Scalar const, 
                                      typename Matrix::Scalar
                                      >::type,
             1>
>::type
#else
/**
 *  @ingroup EigenGroup
 *  @brief Create a 1D Array view into an Eigen object.
 *
 *  The created Array does not own a reference to its data, so the user is responsible for 
 *  ensuring the memory remains valid for the lifetime of the array.
 */
ArrayRef<typename Matrix::Scalar,1>
#endif
viewVectorAsArray(Matrix & matrix) {
    return external(
        const_cast<typename Matrix::Scalar*>(matrix.data()),
        makeVector(matrix.size()),
        makeVector(1)
    );
}

#ifndef DOXYGEN
template <typename Matrix>
typename boost::enable_if_c< 
    Matrix::Flags & Eigen::DirectAccessBit, 
    Array<typename boost::mpl::if_<boost::is_const<Matrix>,
                                   typename Matrix::Scalar const, 
                                   typename Matrix::Scalar
                                   >::type,
          2>
>::type
#else
/**
 *  @ingroup EigenGroup
 *  @brief Create a 2D Array view into an Eigen object.
 *
 *  The created Array does not own a reference to its data, so the user is responsible for 
 *  ensuring the memory remains valid for the lifetime of the array.
 */
ArrayRef<typename Matrix::Scalar,2>
#endif
viewMatrixAsArray(Matrix & matrix) {
    if (Matrix::Flags & Eigen::RowMajorBit) {
        return external(
            const_cast<typename Matrix::Scalar*>(matrix.data()),
            makeVector(matrix.rows(),matrix.cols()),
            makeVector(matrix.stride(),1)
        );
    } else {
        return external(
            const_cast<typename Matrix::Scalar*>(matrix.data()),
            makeVector(matrix.rows(),matrix.cols()),
            makeVector(1,matrix.stride())
        );        
    }
}

}} // namespace lsst::ndarray

#endif // !LSST_NDARRAY_EIGEN_EigenAsArray_h_INCLUDED
