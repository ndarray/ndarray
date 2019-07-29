// -*- c++ -*-
/*
 * Copyright (c) 2010-2012, Jim Bosch
 * All rights reserved.
 *
 * ndarray is distributed under a simple BSD-like license;
 * see the LICENSE file that should be present in the root
 * of the source distribution, or alternately available at:
 * https://github.com/ndarray/ndarray
 */
#include "ndarray/eigen.h"
#include "Eigen/SVD"

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE ndarray-eigen
#include "boost/test/unit_test.hpp"

template <typename T, typename U>
void testElements2(T const & a, U const & b) {
    BOOST_CHECK( a.template getSize<0>() == b.rows() );
    BOOST_CHECK( a.template getSize<1>() == b.cols() );
    BOOST_CHECK( a.template getStride<0>() == b.rowStride() );
    BOOST_CHECK( a.template getStride<1>() == b.colStride() );
    for (int i = 0; i < b.rows(); ++i) {
        for (int j = 0; j < b.cols(); ++j) {
            BOOST_CHECK(&a[i][j] == &b.coeffRef(i,j));
        }
    }    
}

template <typename T, typename U>
void testElements1(T const & a, U const & b) {
    BOOST_CHECK( a.template getSize<0>() == b.size() );
    BOOST_CHECK( a.template getStride<0>() == b.innerStride() );
    for (int i = 0; i < b.size(); ++i) {
        BOOST_CHECK(&a[i] == &b.coeffRef(i));
    }
}

template <typename T, int N, int C>
ndarray::Array<T, N, C> makeMutable(ndarray::Array<T, N, C> const & a) {
    return a;
}

template <typename T, int N, int C>
ndarray::ArrayRef<T, N, C> makeMutable(ndarray::ArrayRef<T, N, C> const & a) {
    return a;
}

template <typename T, int N, int C>
ndarray::Array<T, N, C> makeMutable(ndarray::Array<T const, N, C> const & a) {
    return ndarray::const_array_cast<T>(a);
}

template <typename T, int N, int C>
ndarray::ArrayRef<T, N, C> makeMutable(ndarray::ArrayRef<T const, N, C> const & a) {
    return ndarray::const_array_cast<T>(a.shallow()).deep();
}

template <int Rows, typename XprKind, typename A>
void testAsEigen1(A const & a) {
    typedef typename boost::remove_const<typename A::Element>::type T;
    ndarray::asEigen<XprKind>(makeMutable(a)).setRandom();
    int const n = asEigenMatrix(a).size();
    testElements1(a, ndarray::asEigen<XprKind>(a));
    Eigen::Matrix<T, Rows, Eigen::Dynamic> m1(n, 6);
    m1.setRandom(n, 6);
    Eigen::Matrix<T, Eigen::Dynamic, 1> m2(6, 1);
    m2.setRandom(6, 1);
    asEigenMatrix(makeMutable(a)) = m1 * m2;
    Eigen::Matrix<T, Rows, 1> m3 = m1 * m2;
    for (int i = 0; i < n; ++i) {
        BOOST_CHECK(a[i] == m3(i));
    }
    Eigen::Array<T, Rows, 1> m4(n, 1);
    m4.setRandom();
    Eigen::Array<T, Rows, 1> m5 = m4 * asEigenArray(a);
    Eigen::Array<T, Rows, 1> m6 = m4 * m3.array();
    BOOST_CHECK((m5 == m6).all());
}

template <int Rows, int Cols, typename XprKind, typename A>
void testAsEigen2(A const & a) {
    typedef typename boost::remove_const<typename A::Element>::type T;
    ndarray::asEigen<XprKind>(makeMutable(a)).setRandom();
    int const m = asEigenMatrix(a).rows();
    int const n = asEigenMatrix(a).cols();
    testElements2(a, ndarray::asEigen<XprKind>(a));
    Eigen::Matrix<T, Rows, Eigen::Dynamic> m1(m, 6);
    m1.setRandom(m, 6);
    Eigen::Matrix<T, Eigen::Dynamic, Cols> m2(6, n);
    m2.setRandom(6, n);
    asEigenMatrix(makeMutable(a)) = m1 * m2;
    Eigen::Matrix<T, Rows, Cols> m3 = m1 * m2;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            BOOST_CHECK(a[i][j] == m3(i, j));
        }
    }
    Eigen::Array<T, Rows, Cols> m4(m, n);
    m4.setRandom();
    Eigen::Array<T, Rows, Cols> m5 = m4 * asEigenArray(a);
    Eigen::Array<T, Rows, Cols> m6 = m4 * m3.array();
    BOOST_CHECK((m5 == m6).all());
}

template <typename T, typename XprKind>
void invokeAsEigenTests() {
    ndarray::Array<T, 2, 2> a22(ndarray::allocate(5, 4));
    testAsEigen2<5, 4, XprKind>(a22);

    ndarray::Array<T, 2, 1> a21(a22[ndarray::view()(0, 3)]);
    testAsEigen2<5, 3, XprKind>(a21);
    testAsEigen2<3, 5, XprKind>(a21.transpose());

    ndarray::Array<T, 2, 0> a20(a22[ndarray::view()(0, 4, 2)]);
    testAsEigen2<5, 2, XprKind>(a20);
    testAsEigen2<2, 5, XprKind>(a20.transpose());

    ndarray::Array<T, 1, 1> a11(ndarray::allocate(4));
    testAsEigen1<4, XprKind>(a11);
    testAsEigen1<4, XprKind>(a11.transpose());
    ndarray::Array<T, 1, 0> a10(a11[ndarray::view(0, 4, 2)]);
    testAsEigen1<2, XprKind>(a10);
    testAsEigen1<2, XprKind>(a10.transpose());
}

BOOST_AUTO_TEST_CASE(AsEigen) {
    invokeAsEigenTests<double, Eigen::ArrayXpr>();
    invokeAsEigenTests<double, Eigen::MatrixXpr>();
    invokeAsEigenTests<float const, Eigen::ArrayXpr>();
    invokeAsEigenTests<float const, Eigen::MatrixXpr>();
}

template <typename SVD, typename Matrix, typename Vector>
void testSVD(Matrix const & a, Vector const & b, Vector & x) {
    SVD svd(a, Eigen::ComputeThinU | Eigen::ComputeThinV);
    x = svd.solve(b);
    BOOST_CHECK((a.transpose() * a * x).isApprox(a.transpose() * b));
}

BOOST_AUTO_TEST_CASE(SVD) {
    auto aArray = ndarray::Array<double, 2, 2>(ndarray::allocate(8,5));
    auto bArray = ndarray::Array<double, 1, 1>(ndarray::allocate(8));
    auto xArray = ndarray::Array<double, 1, 1>(ndarray::allocate(5));
    auto a = ndarray::asEigenMatrix(aArray);
    auto b = ndarray::asEigenMatrix(bArray);
    auto x = ndarray::asEigenMatrix(xArray);
    a.setRandom();
    b.setRandom();
    testSVD<Eigen::JacobiSVD<Eigen::MatrixXd>>(a, b, x);
}
