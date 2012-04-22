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
#include "ndarray/eigen/bp/auto.h"

namespace bp = boost::python;
namespace bn = boost::numpy;

template <typename M>
bool acceptMatrix(M m) {
    return (m(0,0) == 1) && (m(0,1) == 2) && (m(0,2) == 3) 
        && (m(1,0) == 4) && (m(1,1) == 5) && (m(1,2) == 6);
}

template <typename M>
bool acceptVector(M m) {
    return (m[0] == 1) && (m[1] == 2) && (m[2] == 3) && (m[3] == 4);
}


template <typename M>
void fillMatrix(M & m) {
    m(0,0) = 1;
    m(0,1) = 2;
    m(0,2) = 3;
    m(1,0) = 4;
    m(1,1) = 5;
    m(1,2) = 6;
}

template <typename M>
M returnEigenView() {
    static typename boost::remove_const<typename boost::remove_reference<M>::type>::type m(
        ndarray::allocate(2,3)
    );
    fillMatrix(m);
    return m;
}

template <typename M>
M returnMatrix() {
    static typename boost::remove_const<typename boost::remove_reference<M>::type>::type m(2,3);
    fillMatrix(m);
    return m;
}

template <typename M>
void fillVector(M & m) {
    m[0] = 1;
    m[1] = 2;
    m[2] = 3;
    m[3] = 4;
}

template <typename M>
M returnVector() {
    static typename boost::remove_const<typename boost::remove_reference<M>::type>::type m(4);
    fillVector(m);
    return m;
}

template <typename M>
bp::object returnObject() {
    static typename boost::remove_const<typename boost::remove_reference<M>::type>::type m(2,3);
    fillMatrix(m);
    bp::object o(m);
    return o;
}

template <typename M>
class MatrixOwner {
    M _matrix;
public:
    MatrixOwner() : _matrix(2,3) { fillMatrix(_matrix); }

    M const & getMatrix_cref() const { return _matrix; }
    M & getMatrix_ref() { return _matrix; }

    bool compareData(bn::matrix const & mp) const {
        return _matrix.data() == reinterpret_cast<double const*>(mp.get_data());
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    static void declare(char const * name) {
        bp::class_< MatrixOwner >(name)
            .def("getMatrix_cref", &MatrixOwner::getMatrix_cref,
                 ndarray::ReturnInternal<>())
            .def("getMatrix_ref", &MatrixOwner::getMatrix_ref,
                 ndarray::ReturnInternal<>())
            .def("compareData", &MatrixOwner::compareData)
            ;
    }
};

static const int X = Eigen::Dynamic;

template <typename T, typename XprKind, int Rows, int Cols>
void wrapEigenView(std::string const & name) {
    bp::def(("accept" + name + "_2p2").c_str(),
            acceptMatrix<ndarray::EigenView<T,2,2,XprKind,Rows,Cols> const &>);
    bp::def(("accept" + name + "_2p1").c_str(),
            acceptMatrix<ndarray::EigenView<T,2,1,XprKind,Rows,Cols> const &>);
    bp::def(("accept" + name + "_2p0").c_str(),
            acceptMatrix<ndarray::EigenView<T,2,0,XprKind,Rows,Cols> const &>);
    bp::def(("accept" + name + "_2m1").c_str(),
            acceptMatrix<ndarray::EigenView<T,2,-1,XprKind,Rows,Cols> const &>);
    bp::def(("accept" + name + "_2m2").c_str(),
            acceptMatrix<ndarray::EigenView<T,2,-2,XprKind,Rows,Cols> const &>);
    bp::def(("return" + name + "_2p2").c_str(),
            returnEigenView<ndarray::EigenView<T,2,2,XprKind,Rows,Cols> >);
    bp::def(("return" + name + "_2p1").c_str(),
            returnEigenView<ndarray::EigenView<T,2,1,XprKind,Rows,Cols> >);
    bp::def(("return" + name + "_2p0").c_str(),
            returnEigenView<ndarray::EigenView<T,2,0,XprKind,Rows,Cols> >);
    bp::def(("return" + name + "_2m1").c_str(),
            returnEigenView<ndarray::EigenView<T,2,-1,XprKind,Rows,Cols> >);
    bp::def(("return" + name + "_2m2").c_str(),
            returnEigenView<ndarray::EigenView<T,2,-2,XprKind,Rows,Cols> >);
}

BOOST_PYTHON_MODULE(eigen_bp_test_mod) {
    bn::initialize();
    wrapEigenView<double,Eigen::MatrixXpr,2,3>("EigenView_M23d");
    wrapEigenView<double,Eigen::MatrixXpr,X,3>("EigenView_MX3d");
    wrapEigenView<double,Eigen::MatrixXpr,2,X>("EigenView_M2Xd");
    wrapEigenView<double,Eigen::MatrixXpr,X,X>("EigenView_MXXd");
    wrapEigenView<int,Eigen::MatrixXpr,2,3>("EigenView_M23i");
    wrapEigenView<int,Eigen::MatrixXpr,X,3>("EigenView_MX3i");
    wrapEigenView<int,Eigen::MatrixXpr,2,X>("EigenView_M2Xi");
    wrapEigenView<int,Eigen::MatrixXpr,X,X>("EigenView_MXXi");

    wrapEigenView<double,Eigen::ArrayXpr,2,3>("EigenView_A23d");
    wrapEigenView<double,Eigen::ArrayXpr,X,3>("EigenView_AX3d");
    wrapEigenView<double,Eigen::ArrayXpr,2,X>("EigenView_A2Xd");
    wrapEigenView<double,Eigen::ArrayXpr,X,X>("EigenView_AXXd");
    wrapEigenView<int,Eigen::ArrayXpr,2,3>("EigenView_A23i");
    wrapEigenView<int,Eigen::ArrayXpr,X,3>("EigenView_AX3i");
    wrapEigenView<int,Eigen::ArrayXpr,2,X>("EigenView_A2Xi");
    wrapEigenView<int,Eigen::ArrayXpr,X,X>("EigenView_AXXi");

    bp::def("acceptMatrix_23d_cref", acceptMatrix< Eigen::Matrix<double,2,3> const & >);
    bp::def("acceptMatrix_X3d_cref", acceptMatrix< Eigen::Matrix<double,X,3> const & >);
    bp::def("acceptMatrix_2Xd_cref", acceptMatrix< Eigen::Matrix<double,2,X> const & >);
    bp::def("acceptMatrix_XXd_cref", acceptMatrix< Eigen::Matrix<double,X,X> const & >);
    bp::def("acceptVector_41d_cref", acceptVector< Eigen::Matrix<double,4,1> const & >);
    bp::def("acceptVector_X1d_cref", acceptVector< Eigen::Matrix<double,X,1> const & >);
    bp::def("acceptVector_14d_cref", acceptVector< Eigen::Matrix<double,1,4> const & >);
    bp::def("acceptVector_1Xd_cref", acceptVector< Eigen::Matrix<double,1,X> const & >);
    bp::def("returnVector_41d", returnVector< Eigen::Matrix<double,4,1> >);
    bp::def("returnVector_14d", returnVector< Eigen::Matrix<double,1,4> >);
    bp::def("returnVector_X1d", returnVector< Eigen::Matrix<double,X,1> >);
    bp::def("returnVector_1Xd", returnVector< Eigen::Matrix<double,1,X> >);
    bp::def("returnMatrix_23d", returnMatrix< Eigen::Matrix<double,2,3> >);
    bp::def("returnMatrix_X3d", returnMatrix< Eigen::Matrix<double,X,3> >);
    bp::def("returnMatrix_2Xd", returnMatrix< Eigen::Matrix<double,2,X> >);
    bp::def("returnMatrix_XXd", returnMatrix< Eigen::Matrix<double,X,X> >);
    bp::def("returnMatrix_23d_c", returnMatrix< Eigen::Matrix<double,2,3> const>);
    bp::def("returnMatrix_X3d_c", returnMatrix< Eigen::Matrix<double,X,3> const>);
    bp::def("returnMatrix_2Xd_c", returnMatrix< Eigen::Matrix<double,2,X> const>);
    bp::def("returnMatrix_XXd_c", returnMatrix< Eigen::Matrix<double,X,X> const>);
    bp::def("returnObject_23d", returnObject< Eigen::Matrix<double,2,3> >);
    bp::def("returnObject_X3d", returnObject< Eigen::Matrix<double,X,3> >);
    bp::def("returnObject_2Xd", returnObject< Eigen::Matrix<double,2,X> >);
    bp::def("returnObject_XXd", returnObject< Eigen::Matrix<double,X,X> >);
    MatrixOwner< Eigen::Matrix<double,2,3> >::declare("MatrixOwner_23d");
    MatrixOwner< Eigen::Matrix<double,X,3> >::declare("MatrixOwner_X3d");
    MatrixOwner< Eigen::Matrix<double,2,X> >::declare("MatrixOwner_2Xd");
    MatrixOwner< Eigen::Matrix<double,X,X> >::declare("MatrixOwner_XXd");
}
