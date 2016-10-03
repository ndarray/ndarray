#include "pybind11/pybind11.h"

#include "numpy/arrayobject.h"
#include "ndarray/pybind11.h"
#include "ndarray/converter.h"

namespace py = pybind11;

Eigen::MatrixXd returnMatrixXd() {
        Eigen::MatrixXd r(5, 3);
            for (int n = 0; n < r.size(); ++n) {
                        r.data()[n] = n;
                            }
                return r;
}

Eigen::Matrix2d returnMatrix2d() {
        Eigen::Matrix2d r;
            for (int n = 0; n < r.size(); ++n) {
                        r.data()[n] = n;
                            }   
                return r;
}

ndarray::Array<double,1,1> returnArray1() {
        ndarray::Array<double,1,1> r(ndarray::allocate(ndarray::makeVector(6)));
            for (int n = 0; n < r.getSize<0>(); ++n) {
                        r[n] = n;
                            }   
                return r;
}

ndarray::Array<double const,1,1> returnConstArray1() {
        return returnArray1();
}

ndarray::Array<double,3> returnArray3() {
        ndarray::Array<double,3,3> r(ndarray::allocate(ndarray::makeVector(4,3,2)));
            ndarray::Array<double,1,1> f = ndarray::flatten<1>(r);
                for (int n = 0; n < f.getSize<0>(); ++n) {
                            f[n] = n;
                                }   
                    return r;
}

ndarray::Array<double const,3> returnConstArray3() {
        return returnArray3();
}

bool acceptMatrixXd(Eigen::MatrixXd const & m1) {
        Eigen::MatrixXd m2 = returnMatrixXd();
            return m1 == m2; 
}

bool acceptMatrix2d(Eigen::Matrix2d const & m1) {
        Eigen::Matrix2d m2 = returnMatrix2d();
            return m1 == m2; 
}

bool acceptArray1(ndarray::Array<double,1,1> const & a1) {
    ndarray::Array<double,1,1> a2 = returnArray1();
#ifndef GCC_45
    return ndarray::all(ndarray::equal(a1, a2));
#else
    return std::equal(a1.begin(), a1.end(), a2.begin());
#endif
}

void acceptArray10(ndarray::Array<double,1,0> const & a1) {}

bool acceptArray3(ndarray::Array<double,3> const & a1) {
    ndarray::Array<double,3> a2 = returnArray3();
#ifndef GCC_45
    return ndarray::all(ndarray::equal(a1, a2));
#else
    for (int i = 0; i < a1.getSize<0>(); ++i) {
      for (int j = 0; j < a1.getSize<1>(); ++j) {
        if (!std::equal(a1[i][j].begin(), a1[i][j].end(), a2[i][j].begin())) return false;
      }
    }
    return true;
#endif
}

int acceptOverload(int n) {
    return 0;
}

int acceptOverload(Eigen::Matrix3d const & m) {
    return 3;
}

int acceptOverload(Eigen::Matrix2d const & m) {
    return 2;
}

struct MatrixOwner {
    typedef Eigen::Matrix<double,2,2,Eigen::DontAlign> MemberMatrix;
    MemberMatrix member;
    MemberMatrix & getMember() { return member; }
    explicit MatrixOwner() : member(MemberMatrix::Zero()) {}
};

PYBIND11_PLUGIN(pybind11_test_mod) {
    pybind11::module mod("pybind11_test_mod", "Tests for the ndarray library");

    if (_import_array() < 0) {
        PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
        return nullptr;
    }

    py::class_<MatrixOwner> cls(mod, "MatrixOwner");
    cls.def(py::init<>());

    py::class_<MatrixOwner::MemberMatrix>(cls, "MemberMatrix");

    cls.def_readwrite("member", &MatrixOwner::member);
    cls.def("getMember", &MatrixOwner::getMember);
    mod.def("returnMatrixXd", returnMatrixXd);
    mod.def("returnMatrix2d", returnMatrix2d);
    mod.def("returnArray1", returnArray1);
    mod.def("returnConstArray1", returnConstArray1);
    mod.def("returnArray3", returnArray3);
    mod.def("returnConstArray3", returnConstArray3);
    mod.def("acceptMatrixXd", acceptMatrixXd);
    mod.def("acceptMatrix2d", acceptMatrix2d);
    mod.def("acceptArray1", acceptArray1);
    mod.def("acceptArray10", acceptArray10);
    mod.def("acceptArray3", acceptArray3);
    mod.def("acceptOverload", (int (*)(int)) acceptOverload);
    mod.def("acceptOverload", (int (*)(Eigen::Matrix2d const &)) acceptOverload);
    mod.def("acceptOverload", (int (*)(Eigen::Matrix3d const &)) acceptOverload);

    return mod.ptr();
}