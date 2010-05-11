#include "lsst/ndarray/eigen.hpp"

#include <Eigen/Array>

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE ndarray-eigen
#include <boost/test/unit_test.hpp>

namespace ndarray = lsst::ndarray;

BOOST_AUTO_TEST_CASE(MatrixXd) {
    Eigen::MatrixXd m1 = Eigen::MatrixXd::Random(5,6);
    ndarray::Array<double,2> a1(ndarray::viewMatrixAsArray(m1));
    BOOST_CHECK_EQUAL(m1.data(),a1.getData());
    BOOST_CHECK_EQUAL(m1.rows(),a1.getSize<0>());
    BOOST_CHECK_EQUAL(m1.cols(),a1.getSize<1>());
    Eigen::Block<Eigen::MatrixXd> m3 = m1.block(0,1,3,3);
    ndarray::Array<double,2> a3(ndarray::viewMatrixAsArray(m3));
    BOOST_CHECK_EQUAL(m3.data(),a3.getData());
    BOOST_CHECK_EQUAL(m3.rows(),a3.getSize<0>());
    BOOST_CHECK_EQUAL(m3.cols(),a3.getSize<1>());
}

BOOST_AUTO_TEST_CASE(Matrix3d) {
    Eigen::Matrix3d m1 = Eigen::Matrix3d::Random();
    ndarray::Array<double,2> a1(ndarray::viewMatrixAsArray(m1));
    BOOST_CHECK_EQUAL(m1.data(),a1.getData());
    BOOST_CHECK_EQUAL(m1.rows(),a1.getSize<0>());
    BOOST_CHECK_EQUAL(m1.cols(),a1.getSize<1>());
    ndarray::Array<double,2,2> a2(ndarray::copy(a1));
    {
        Eigen::Block<Eigen::Matrix3d,2,2> m3 = m1.block<2,2>(0,0);
        ndarray::Array<double,2> a3(ndarray::viewMatrixAsArray(m3));
        ndarray::Array<double,2,2> a4(ndarray::copy(a3));
        BOOST_CHECK_EQUAL(m3.data(),a3.getData());
        BOOST_CHECK_EQUAL(m3.rows(),a3.getSize<0>());
        BOOST_CHECK_EQUAL(m3.cols(),a3.getSize<1>());
    }
    {
        Eigen::Block<Eigen::Matrix3d,2,1> m3 = m1.block<2,1>(0,2);
        ndarray::Array<double,1> a3(ndarray::viewVectorAsArray(m3));
        ndarray::Array<double,1,1> a4(ndarray::copy(a3));
        BOOST_CHECK_EQUAL(m3.data(),a3.getData());
        BOOST_CHECK_EQUAL(m3.rows(),a3.getSize<0>());
        BOOST_CHECK_EQUAL(m3.cols(),a3.getSize<1>());
    }    
}

BOOST_AUTO_TEST_CASE(VectorXd) {
    Eigen::VectorXd m1 = Eigen::VectorXd::Random(5);
    ndarray::Array<double,2> a1(ndarray::viewMatrixAsArray(m1));
    BOOST_CHECK_EQUAL(m1.data(),a1.getData());
    BOOST_CHECK_EQUAL(m1.rows(),a1.getSize<0>());
    BOOST_CHECK_EQUAL(m1.cols(),a1.getSize<1>());
    ndarray::Array<double,1> a3(ndarray::viewVectorAsArray(m1));
    BOOST_CHECK_EQUAL(m1.data(),a1.getData());
    BOOST_CHECK_EQUAL(m1.rows(),a1.getSize<0>());
}

BOOST_AUTO_TEST_CASE(Vector3d) {
    Eigen::Vector3d m1 = Eigen::Vector3d::Random();
    ndarray::Array<double,2> a1(ndarray::viewMatrixAsArray(m1));
    BOOST_CHECK_EQUAL(m1.data(),a1.getData());
    BOOST_CHECK_EQUAL(m1.rows(),a1.getSize<0>());
    BOOST_CHECK_EQUAL(m1.cols(),a1.getSize<1>());
    ndarray::Array<double,1> a3(ndarray::viewVectorAsArray(m1));
    BOOST_CHECK_EQUAL(m1.data(),a1.getData());
    BOOST_CHECK_EQUAL(m1.rows(),a1.getSize<0>());
}

template <int C>
void testEigenView2() {
    ndarray::Vector<int,2> shape = ndarray::makeVector(5,4);

    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor|Eigen::AutoAlign> 
        mr(Eigen::MatrixXd::Random(shape[0],shape[1]));
    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::AutoAlign> 
        mc(Eigen::MatrixXd::Random(shape[1],shape[0]));

    ndarray::Array<double,2,C> a(ndarray::allocate(shape));

    ndarray::EigenView<double,2,C> vr(a);
    vr = mr;
    BOOST_CHECK_EQUAL(vr,mr);
    mr.setConstant(0.0);
    mr = vr;
    BOOST_CHECK_EQUAL(ndarray::viewAsEigen(a),mr);

    ndarray::TransposedEigenView<double,2,C> vc(a);
    vc = mc;
    BOOST_CHECK_EQUAL(vc,mc);
    mc.setConstant(0.0);
    mc = vc;
    BOOST_CHECK_EQUAL(ndarray::viewAsTransposedEigen(a),mc);
}

template <int C>
void testEigenView1() {
    ndarray::Vector<int,1> shape = ndarray::makeVector(5);

    Eigen::VectorXd mr(Eigen::VectorXd::Random(shape[0]));
    Eigen::RowVectorXd mc(Eigen::RowVectorXd::Random(shape[0]));
    ndarray::Array<double,1,C> a(ndarray::allocate(shape));

    ndarray::EigenView<double,1,C> vr(a);
    vr = mr;
    BOOST_CHECK_EQUAL(vr,mr);
    mr.setConstant(0.0);
    mr = vr;
    BOOST_CHECK_EQUAL(ndarray::viewAsEigen(a),mr);

    ndarray::TransposedEigenView<double,1,C> vc(a);
    vc = mc;
    BOOST_CHECK_EQUAL(vc,mc);
    mc.setConstant(0.0);
    mc = vc;
    BOOST_CHECK_EQUAL(ndarray::viewAsTransposedEigen(a),mc);
}

BOOST_AUTO_TEST_CASE(EigenView) {
    
    testEigenView2<0>();
    testEigenView2<1>();
    testEigenView2<2>();

    testEigenView1<0>();
    testEigenView1<1>();

}

BOOST_AUTO_TEST_CASE(ArrayOfVector) {
    ndarray::Array<Eigen::Vector4d,2,2> a1 = 
        ndarray::allocate< Eigen::aligned_allocator<double> >(ndarray::makeVector(5,4));
    a1.deep() = Eigen::Vector4d::Zero();
};
