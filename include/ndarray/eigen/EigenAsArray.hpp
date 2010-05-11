#ifndef NDARRAY_EIGEN_EigenAsArray_hpp_INCLUDED
#define NDARRAY_EIGEN_EigenAsArray_hpp_INCLUDED

/**
 *  @file ndarray/eigen/EigenAsArray.hpp
 *  @brief Functions that construct ndarray::Array views into Eigen objects.
 *
 *  \note This file is not included by the main "ndarray.hpp" header file.
 */

#include "ndarray/eigen_fwd.hpp"
#include "ndarray.hpp"
#include <Eigen/Core>

namespace ndarray {

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

} // namespace ndarray

#endif // !NDARRAY_EIGEN_EigenAsArray_hpp_INCLUDED
