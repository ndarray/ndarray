#ifndef LSST_NDARRAY_EIGEN_ArrayAsEigen_hpp_INCLUDED
#define LSST_NDARRAY_EIGEN_ArrayAsEigen_hpp_INCLUDED

/**
 *  @file lsst/ndarray/eigen/ArrayAsEigen.hpp
 *  @brief Custom Eigen matrix objects that present a view into an lsst::ndarray::Array.
 *
 *  \note This file is not included by the main "lsst/ndarray.hpp" header file.
 */

#include "lsst/ndarray.hpp"
#include "lsst/ndarray/eigen_fwd.hpp"
#include <Eigen/Core>

#ifndef DOXYGEN
namespace Eigen {

template <typename T, int C>
struct ei_traits< lsst::ndarray::EigenView<T,2,C> >{
    typedef typename boost::remove_const<T>::type Scalar;
    enum {
        RowsAtCompileTime = Dynamic,
        ColsAtCompileTime = Dynamic,
        MaxRowsAtCompileTime = Dynamic,
        MaxColsAtCompileTime = Dynamic,
        Flags = RowMajorBit | ((C>0) ? (DirectAccessBit|PacketAccessBit):0),
        CoeffReadCost = NumTraits<Scalar>::ReadCost
    };
};

template <typename T, int C>
struct ei_traits< lsst::ndarray::EigenView<T,1,C> >{
    typedef typename boost::remove_const<T>::type Scalar;
    enum {
        RowsAtCompileTime = Eigen::Dynamic,
        ColsAtCompileTime = 1,
        MaxRowsAtCompileTime = Eigen::Dynamic,
        MaxColsAtCompileTime = 1,
        Flags = LinearAccessBit | DirectAccessBit | ((C==0) ? RowMajorBit:PacketAccessBit),
        CoeffReadCost = NumTraits<Scalar>::ReadCost
    };
};

template <typename T, int C>
struct ei_traits< lsst::ndarray::TransposedEigenView<T,2,C> >{
    typedef typename boost::remove_const<T>::type Scalar;
    enum {
        RowsAtCompileTime = Dynamic,
        ColsAtCompileTime = Dynamic,
        MaxRowsAtCompileTime = Dynamic,
        MaxColsAtCompileTime = Dynamic,
        Flags = ((C>0) ? (DirectAccessBit|PacketAccessBit):0),
        CoeffReadCost = NumTraits<Scalar>::ReadCost
    };
};

template <typename T, int C>
struct ei_traits< lsst::ndarray::TransposedEigenView<T,1,C> >{
    typedef typename boost::remove_const<T>::type Scalar;
    enum {
        RowsAtCompileTime = 1,
        ColsAtCompileTime = Eigen::Dynamic,
        MaxRowsAtCompileTime = 1,
        MaxColsAtCompileTime = Eigen::Dynamic,
        Flags = LinearAccessBit | DirectAccessBit | ((C==0) ? 0:(RowMajorBit|PacketAccessBit)),
        CoeffReadCost = NumTraits<Scalar>::ReadCost
    };
};

} // namespace Eigen
#endif // DOXYGEN

namespace lsst { namespace ndarray {

template <typename T, int N, int C>
class EigenView {
    BOOST_STATIC_ASSERT(sizeof(T)<0); // Only 1d and 2d views allowed.
};

template <typename T, int N, int C>
class TransposedEigenView {
    BOOST_STATIC_ASSERT(sizeof(T)<0); // Only 1d and 2d views allowed.
};

/**
 *  @ingroup EigenGroup
 *  @brief A Eigen matrix view into a 2D Array.
 *
 *  See the Eigen::MatrixBase documentation for more information.
 */
template <typename T, int C>
class EigenView<T,2,C> : public Eigen::MatrixBase< EigenView<T,2,C> > {
public:

    EIGEN_GENERIC_PUBLIC_INTERFACE(EigenView);

    explicit EigenView(Array<T,2,C> const & array) : _array(array) {}

    EIGEN_INHERIT_ASSIGNMENT_OPERATORS(EigenView);

    T * data() { return _array.getData(); }
    Scalar const * data() const { return _array.getData(); }

    inline int rows() const { return _array.template getSize<0>(); }
    inline int cols() const { return _array.template getSize<1>(); }

    inline int stride() const { return _array.template getStride<0>(); }

    inline T & coeffRef(int row, int col) { return _array[makeVector(row,col)]; }
    inline Scalar const coeff(int row, int col) const { return _array[makeVector(row,col)]; }

    inline T & coeffRef(int index) { return _array[makeVector(index,0)]; }
    inline Scalar const coeff(int index) const { return _array[makeVector(index,0)]; }

    template<int LoadMode>
    inline PacketScalar packet(int row, int col) const {
        return Eigen::ei_ploadt<Scalar,LoadMode>(&_array[makeVector(row,col)]);
    }

    template<int StoreMode>
    inline void writePacket(int row, int col, const PacketScalar& x) {
        Eigen::ei_pstoret<Scalar,PacketScalar,StoreMode>(&_array[makeVector(row,col)], x);
    }

    template<int LoadMode>
    inline PacketScalar packet(int index) const {
        return Eigen::ei_ploadt<Scalar,LoadMode>(&_array[makeVector(index,0)]);
    }

    template<int StoreMode>
    inline void writePacket(int index, const PacketScalar& x) {
        Eigen::ei_pstoret<Scalar,PacketScalar,StoreMode>(&_array[makeVector(index,0)], x);
    }

    Array<T,2,C> const & getArray() const { return _array; }

private:
    Array<T,2,C> _array;
};


/**
 *  @ingroup EigenGroup
 *  @brief An Eigen vector view into a 1D Array.
 *
 *  See the Eigen::MatrixBase documentation for more information.
 */
template <typename T, int C>
class EigenView<T,1,C> : public Eigen::MatrixBase< EigenView<T,1,C> > {
public:

    EIGEN_GENERIC_PUBLIC_INTERFACE(EigenView);

    explicit EigenView(Array<T,1,C> const & array) : _array(array) {}

    EIGEN_INHERIT_ASSIGNMENT_OPERATORS(EigenView);

    T * data() { return _array.getData(); }
    Scalar const * data() const { return _array.getData(); }

    inline int rows() const { return _array.template getSize<0>(); }
    inline int cols() const { return 1; }

    inline int stride() const {
        return (C==1) ? _array.template getStride<1>() : _array.template getStride<0>();
    }

    inline T & coeffRef(int row, int) { return _array[row]; }
    inline Scalar const coeff(int row, int) const { return _array[row]; }

    inline T & coeffRef(int index) { return _array[index]; }
    inline Scalar const coeff(int index) const { return _array[index]; }

    template<int LoadMode>
    inline PacketScalar packet(int row, int) const {
        return Eigen::ei_ploadt<Scalar,LoadMode>(&_array[row]);
    }

    template<int StoreMode>
    inline void writePacket(int row, int, const PacketScalar& x) {
        Eigen::ei_pstoret<Scalar,PacketScalar,StoreMode>(&_array[row], x);
    }

    template<int LoadMode>
    inline PacketScalar packet(int index) const {
        return Eigen::ei_ploadt<Scalar,LoadMode>(&_array[index]);
    }

    template<int StoreMode>
    inline void writePacket(int index, const PacketScalar& x) {
        Eigen::ei_pstoret<Scalar,PacketScalar,StoreMode>(&_array[index], x);
    }

    Array<T,1,C> const & getArray() const { return _array; }

private:
    Array<T,1,C> _array;
};

/**
 *  @ingroup EigenGroup
 *  @brief An Eigen matrix view into the transpose of a 2D Array.
 *
 *  See the Eigen::MatrixBase documentation for more information.
 */
template <typename T, int C>
class TransposedEigenView<T,2,C> : public Eigen::MatrixBase< TransposedEigenView<T,2,C> > {
public:

    EIGEN_GENERIC_PUBLIC_INTERFACE(TransposedEigenView);

    explicit TransposedEigenView(Array<T,2,C> const & array) : _array(array) {}

    EIGEN_INHERIT_ASSIGNMENT_OPERATORS(TransposedEigenView);

    T * data() { return _array.getData(); }
    Scalar const * data() const { return _array.getData(); }

    inline int rows() const { return _array.template getSize<1>(); }
    inline int cols() const { return _array.template getSize<0>(); }

    inline int stride() const { return _array.template getStride<0>(); }

    inline T & coeffRef(int row, int col) { return _array[makeVector(col,row)]; }
    inline Scalar const coeff(int row, int col) const { return _array[makeVector(col,row)]; }

    inline T & coeffRef(int index) { return _array[makeVector(0,index)]; }
    inline Scalar const coeff(int index) const { return _array[makeVector(0,index)]; }

    template<int LoadMode>
    inline PacketScalar packet(int row, int col) const {
        return Eigen::ei_ploadt<Scalar,LoadMode>(&_array[makeVector(col,row)]);
    }

    template<int StoreMode>
    inline void writePacket(int row, int col, const PacketScalar& x) {
        Eigen::ei_pstoret<Scalar,PacketScalar,StoreMode>(&_array[makeVector(col,row)], x);
    }

    template<int LoadMode>
    inline PacketScalar packet(int index) const {
        return Eigen::ei_ploadt<Scalar,LoadMode>(&_array[makeVector(0,index)]);
    }

    template<int StoreMode>
    inline void writePacket(int index, const PacketScalar& x) {
        Eigen::ei_pstoret<Scalar,PacketScalar,StoreMode>(&_array[makeVector(0,index)], x);
    }

    Array<T,2,C> const & getArray() const { return _array; }

private:
    Array<T,2,C> _array;
};

/**
 *  @ingroup EigenGroup
 *  @brief An Eigen vector view into the transpose of a 1D Array.
 *
 *  See the Eigen::MatrixBase documentation for more information.
 */
template <typename T, int C>
class TransposedEigenView<T,1,C> : public Eigen::MatrixBase< TransposedEigenView<T,1,C> > {
public:

    EIGEN_GENERIC_PUBLIC_INTERFACE(TransposedEigenView);

    explicit TransposedEigenView(Array<T,1,C> const & array) : _array(array) {}

    EIGEN_INHERIT_ASSIGNMENT_OPERATORS(TransposedEigenView);

    T * data() { return _array.getData(); }
    Scalar const * data() const { return _array.getData(); }

    inline int rows() const { return 1; }
    inline int cols() const { return _array.template getSize<0>(); }

    inline int stride() const {
        return (C==1) ? _array.template getStride<1>() : _array.template getStride<0>();
    }

    inline T & coeffRef(int, int col) { return _array[col]; }
    inline Scalar const coeff(int, int col) const { return _array[col]; }

    inline T & coeffRef(int index) { return _array[index]; }
    inline Scalar const coeff(int index) const { return _array[index]; }

    template<int LoadMode>
    inline PacketScalar packet(int, int col) const {
        return Eigen::ei_ploadt<Scalar,LoadMode>(&_array[col]);
    }

    template<int StoreMode>
    inline void writePacket(int, int col, const PacketScalar& x) {
        Eigen::ei_pstoret<Scalar,PacketScalar,StoreMode>(&_array[col], x);
    }

    template<int LoadMode>
    inline PacketScalar packet(int index) const {
        return Eigen::ei_ploadt<Scalar,LoadMode>(&_array[index]);
    }

    template<int StoreMode>
    inline void writePacket(int index, const PacketScalar& x) {
        Eigen::ei_pstoret<Scalar,PacketScalar,StoreMode>(&_array[index], x);
    }

    Array<T,1,C> const & getArray() const { return _array; }

private:
    Array<T,1,C> _array;
};

/**
 *  @ingroup EigenGroup
 *  @brief Return a non-transposed Eigen view of the given 1D or 2D Array.
 */
template <typename T, int N, int C>
inline EigenView<T,N,C>
viewAsEigen(Array<T,N,C> const & array) {
    return EigenView<T,N,C>(array);
}

/**
 *  @ingroup EigenGroup
 *  @brief Return a transposed Eigen view of the given 1D or 2D Array.
 */
template <typename T, int N, int C>
inline TransposedEigenView<T,N,C>
viewAsTransposedEigen(Array<T,N,C> const & array) {
    return TransposedEigenView<T,N,C>(array);
}

}} // namespace lsst::ndarray

#endif // !LSST_NDARRAY_EIGEN_ArrayAsEigen_hpp_INCLUDED
