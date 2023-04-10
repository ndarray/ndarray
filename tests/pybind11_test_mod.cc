#include "pybind11/pybind11.h"

#include "ndarray/pybind11.h"

namespace py = pybind11;
using namespace py::literals;

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

bool acceptArray1(ndarray::Array<double,1,1> const & a1) {
    ndarray::Array<double,1,1> a2 = returnArray1();
#ifndef GCC_45
    return ndarray::all(ndarray::equal(a1, a2));
#else
    return std::equal(a1.begin(), a1.end(), a2.begin());
#endif
}

void acceptArray10(ndarray::Array<double,1,0> const & a1) {}

void acceptArray11(ndarray::Array<double,1,1> const & a1) {}

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

int acceptNoneArray(ndarray::Array<double, 1, 1> const * array = nullptr) {
    if (array) {
        return 0;
    } else {
        return 1;
    }
}

PYBIND11_MODULE(pybind11_test_mod, mod) {
    mod.def("returnArray1", returnArray1);
    mod.def("returnConstArray1", returnConstArray1);
    mod.def("returnArray3", returnArray3);
    mod.def("returnConstArray3", returnConstArray3);
    mod.def("acceptArray1", acceptArray1);
    mod.def("acceptArray10", acceptArray10);
    mod.def("acceptArray11", acceptArray11);
    mod.def("acceptArray3", acceptArray3);
    mod.def("acceptNoneArray", acceptNoneArray, "array"_a = nullptr);
}
