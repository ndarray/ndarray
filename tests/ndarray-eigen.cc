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
#include <lsst/ndarray/eigen.h>

#include <Eigen/Array>

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE ndarray-eigen
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_CASE(MatrixXd) {
    Eigen::MatrixXd m1 = Eigen::MatrixXd::Random(5,6);
    lsst::ndarray::Array<double,2> a1(lsst::ndarray::viewMatrixAsArray(m1));
    BOOST_CHECK_EQUAL(m1.data(),a1.getData());
    BOOST_CHECK_EQUAL(m1.rows(),a1.getSize<0>());
    BOOST_CHECK_EQUAL(m1.cols(),a1.getSize<1>());
    Eigen::Block<Eigen::MatrixXd> m3 = m1.block(0,1,3,3);
    lsst::ndarray::Array<double,2> a3(lsst::ndarray::viewMatrixAsArray(m3));
    BOOST_CHECK_EQUAL(m3.data(),a3.getData());
    BOOST_CHECK_EQUAL(m3.rows(),a3.getSize<0>());
    BOOST_CHECK_EQUAL(m3.cols(),a3.getSize<1>());
}

BOOST_AUTO_TEST_CASE(Matrix3d) {
    Eigen::Matrix3d m1 = Eigen::Matrix3d::Random();
    lsst::ndarray::Array<double,2> a1(lsst::ndarray::viewMatrixAsArray(m1));
    BOOST_CHECK_EQUAL(m1.data(),a1.getData());
    BOOST_CHECK_EQUAL(m1.rows(),a1.getSize<0>());
    BOOST_CHECK_EQUAL(m1.cols(),a1.getSize<1>());
    lsst::ndarray::Array<double,2,2> a2(lsst::ndarray::copy(a1));
    {
        Eigen::Block<Eigen::Matrix3d,2,2> m3 = m1.block<2,2>(0,0);
        lsst::ndarray::Array<double,2> a3(lsst::ndarray::viewMatrixAsArray(m3));
        lsst::ndarray::Array<double,2,2> a4(lsst::ndarray::copy(a3));
        BOOST_CHECK_EQUAL(m3.data(),a3.getData());
        BOOST_CHECK_EQUAL(m3.rows(),a3.getSize<0>());
        BOOST_CHECK_EQUAL(m3.cols(),a3.getSize<1>());
    }
    {
        Eigen::Block<Eigen::Matrix3d,2,1> m3 = m1.block<2,1>(0,2);
        lsst::ndarray::Array<double,1> a3(lsst::ndarray::viewVectorAsArray(m3));
        lsst::ndarray::Array<double,1,1> a4(lsst::ndarray::copy(a3));
        BOOST_CHECK_EQUAL(m3.data(),a3.getData());
        BOOST_CHECK_EQUAL(m3.rows(),a3.getSize<0>());
        BOOST_CHECK_EQUAL(m3.cols(),a3.getSize<1>());
    }    
}

BOOST_AUTO_TEST_CASE(VectorXd) {
    Eigen::VectorXd m1 = Eigen::VectorXd::Random(5);
    lsst::ndarray::Array<double,2> a1(lsst::ndarray::viewMatrixAsArray(m1));
    BOOST_CHECK_EQUAL(m1.data(),a1.getData());
    BOOST_CHECK_EQUAL(m1.rows(),a1.getSize<0>());
    BOOST_CHECK_EQUAL(m1.cols(),a1.getSize<1>());
    lsst::ndarray::Array<double,1> a3(lsst::ndarray::viewVectorAsArray(m1));
    BOOST_CHECK_EQUAL(m1.data(),a1.getData());
    BOOST_CHECK_EQUAL(m1.rows(),a1.getSize<0>());
}

BOOST_AUTO_TEST_CASE(Vector3d) {
    Eigen::Vector3d m1 = Eigen::Vector3d::Random();
    lsst::ndarray::Array<double,2> a1(lsst::ndarray::viewMatrixAsArray(m1));
    BOOST_CHECK_EQUAL(m1.data(),a1.getData());
    BOOST_CHECK_EQUAL(m1.rows(),a1.getSize<0>());
    BOOST_CHECK_EQUAL(m1.cols(),a1.getSize<1>());
    lsst::ndarray::Array<double,1> a3(lsst::ndarray::viewVectorAsArray(m1));
    BOOST_CHECK_EQUAL(m1.data(),a1.getData());
    BOOST_CHECK_EQUAL(m1.rows(),a1.getSize<0>());
}

template <int C>
void testEigenView2() {
    lsst::ndarray::Vector<int,2> shape = lsst::ndarray::makeVector(5,4);

    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor|Eigen::AutoAlign> 
        mr(Eigen::MatrixXd::Random(shape[0],shape[1]));
    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::AutoAlign> 
        mc(Eigen::MatrixXd::Random(shape[1],shape[0]));

    lsst::ndarray::Array<double,2,C> a(lsst::ndarray::allocate(shape));

    lsst::ndarray::EigenView<double,2,C> vr(a);
    vr = mr;
    BOOST_CHECK_EQUAL(vr,mr);
    mr.setConstant(0.0);
    mr = vr;
    BOOST_CHECK_EQUAL(lsst::ndarray::viewAsEigen(a),mr);

    lsst::ndarray::TransposedEigenView<double,2,C> vc(a);
    vc = mc;
    BOOST_CHECK_EQUAL(vc,mc);
    mc.setConstant(0.0);
    mc = vc;
    BOOST_CHECK_EQUAL(lsst::ndarray::viewAsTransposedEigen(a),mc);
}

template <int C>
void testEigenView1() {
    lsst::ndarray::Vector<int,1> shape = lsst::ndarray::makeVector(5);

    Eigen::VectorXd mr(Eigen::VectorXd::Random(shape[0]));
    Eigen::RowVectorXd mc(Eigen::RowVectorXd::Random(shape[0]));
    lsst::ndarray::Array<double,1,C> a(lsst::ndarray::allocate(shape));

    lsst::ndarray::EigenView<double,1,C> vr(a);
    vr = mr;
    BOOST_CHECK_EQUAL(vr,mr);
    mr.setConstant(0.0);
    mr = vr;
    BOOST_CHECK_EQUAL(lsst::ndarray::viewAsEigen(a),mr);

    lsst::ndarray::TransposedEigenView<double,1,C> vc(a);
    vc = mc;
    BOOST_CHECK_EQUAL(vc,mc);
    mc.setConstant(0.0);
    mc = vc;
    BOOST_CHECK_EQUAL(lsst::ndarray::viewAsTransposedEigen(a),mc);
}

BOOST_AUTO_TEST_CASE(EigenView) {
    
    testEigenView2<0>();
    testEigenView2<1>();
    testEigenView2<2>();

    testEigenView1<0>();
    testEigenView1<1>();

}
