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
#include "lsst/ndarray.h"

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE ndarray
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_CASE(sizes) {
    std::cerr << "sizeof(int): " << sizeof(int) << "\n";
    std::cerr << "sizeof(double*): " << sizeof(double*) << "\n";
    std::cerr << "sizeof(shared_ptr<double>): " << sizeof(boost::shared_ptr<double>) << "\n";
    std::cerr << "sizeof(Core<double,1>): " << sizeof(lsst::ndarray::detail::Core<1>) << "\n";
    std::cerr << "sizeof(Core<double,2>): " << sizeof(lsst::ndarray::detail::Core<2>) << "\n";
    std::cerr << "sizeof(Core<double,3>): " << sizeof(lsst::ndarray::detail::Core<3>) << "\n";
}

BOOST_AUTO_TEST_CASE(vectors) {
    lsst::ndarray::Vector<int,3> a = lsst::ndarray::makeVector(5,6,7);
    BOOST_CHECK_EQUAL(a[0],5);
    BOOST_CHECK_EQUAL(a[1],6);
    BOOST_CHECK_EQUAL(a[2],7);
    lsst::ndarray::Vector<int,3> b(a);
    BOOST_CHECK_EQUAL(a,b);
    lsst::ndarray::Vector<double,3> c(a);
    BOOST_CHECK_EQUAL(c[0],5.0);
    BOOST_CHECK_EQUAL(c[1],6.0);
    BOOST_CHECK_EQUAL(c[2],7.0);
    lsst::ndarray::Vector<double,3> d(5.0);
    BOOST_CHECK_EQUAL(d[0],5.0);
    BOOST_CHECK_EQUAL(d[1],5.0);
    BOOST_CHECK_EQUAL(d[2],5.0);

    lsst::ndarray::Vector<int,0> e;
    lsst::ndarray::Vector<int,3> f = lsst::ndarray::concatenate(a, e);
    BOOST_CHECK_EQUAL(a, f);

    e = a.last<0>();
}

BOOST_AUTO_TEST_CASE(cores) {
    typedef lsst::ndarray::detail::Core<3> Core;
    lsst::ndarray::Vector<int,3> shape = lsst::ndarray::makeVector(4,3,2);
    lsst::ndarray::Vector<int,3> strides = lsst::ndarray::makeVector(6,2,1);
    Core::Ptr core = Core::create(shape, strides);
    BOOST_CHECK_EQUAL(core->getRC(),1);
    Core::Ptr copy = core;
    BOOST_CHECK_EQUAL(core->getRC(),2);
    copy.reset();
    BOOST_CHECK_EQUAL(core->getRC(),1);
}

BOOST_AUTO_TEST_CASE(allocation) {
    lsst::ndarray::Vector<int,3> shape = lsst::ndarray::makeVector(5,6,7);
    lsst::ndarray::Array<float,3,3> a = lsst::ndarray::allocate(shape);
    BOOST_CHECK_EQUAL(a.getShape(), shape);
    lsst::ndarray::Array<float,3> b = lsst::ndarray::allocate(shape);
    BOOST_CHECK_EQUAL(b.getShape(), shape);
    BOOST_CHECK_EQUAL(b.getSize<0>(), shape[0]);
    BOOST_CHECK_EQUAL(b.getSize<1>(), shape[1]);
    BOOST_CHECK_EQUAL(b.getSize<2>(), shape[2]);
    BOOST_CHECK_EQUAL(b.getStride<0>(), 6*7);
    BOOST_CHECK_EQUAL(b.getStride<1>(), 7);
    BOOST_CHECK_EQUAL(b.getStride<2>(), 1);
    BOOST_CHECK_EQUAL(b.getStrides(), lsst::ndarray::makeVector(6*7,7,1));
    lsst::ndarray::Array<int,1> c = lsst::ndarray::allocate(lsst::ndarray::makeVector(5));
    BOOST_CHECK_EQUAL(b.size(), 5);
    
}

BOOST_AUTO_TEST_CASE(external) {
    double data[3*4*2] = {0};
    lsst::ndarray::Vector<int,3> shape = lsst::ndarray::makeVector(3,4,2);
    lsst::ndarray::Vector<int,3> strides = lsst::ndarray::makeVector(8,2,1);
    lsst::ndarray::Array<double,3,3> a = lsst::ndarray::external(data,shape,strides);
    BOOST_CHECK_EQUAL(a.getData(), data);
    BOOST_CHECK_EQUAL(a.getShape(), shape);
    BOOST_CHECK_EQUAL(a.getStrides(), strides);
}

BOOST_AUTO_TEST_CASE(conversion) {
    double data[3*4*2] = {0};
    lsst::ndarray::Vector<int,3> shape = lsst::ndarray::makeVector(3,4,2);
    lsst::ndarray::Vector<int,3> strides = lsst::ndarray::makeVector(8,2,1);
    lsst::ndarray::Array<double,3,3> a = lsst::ndarray::external(data,shape,strides);
    lsst::ndarray::Array<double const,3> b = a;
}

BOOST_AUTO_TEST_CASE(shallow) {
    double data[3*4*2] = {0};
    lsst::ndarray::Vector<int,3> shape = lsst::ndarray::makeVector(3,4,2);
    lsst::ndarray::Vector<int,3> strides = lsst::ndarray::makeVector(8,2,1);
    lsst::ndarray::Array<double,3,3> a = lsst::ndarray::external(data,shape,strides);
    lsst::ndarray::Array<double,3,1> b = lsst::ndarray::external(data,shape,strides);
    BOOST_CHECK(a == b);
    BOOST_CHECK(a[2].shallow() == b[2].shallow());
    BOOST_CHECK(a[0][1].shallow() == b[0][1].shallow());
    BOOST_CHECK(a[0][1].shallow() != b[1][2].shallow());
    lsst::ndarray::Array<double,3,3> c;
    c = a;
    BOOST_CHECK_EQUAL(a.getData(), c.getData());
    BOOST_CHECK_EQUAL(a.getShape(), c.getShape());
    BOOST_CHECK_EQUAL(a.getStrides(), c.getStrides());
    BOOST_CHECK(a.shallow() == c.shallow());
    lsst::ndarray::Array<double,2> d = c[1];
    BOOST_CHECK_EQUAL(d.getData(), c[1].getData());
    BOOST_CHECK_EQUAL(d.getShape(), c[1].getShape());
    BOOST_CHECK_EQUAL(d.getStrides(), c[1].getStrides());
    BOOST_CHECK(d.shallow() == c[1].shallow());
}

BOOST_AUTO_TEST_CASE(casts) {
    double data[3*4*2] = {0};
    lsst::ndarray::Vector<int,3> shape = lsst::ndarray::makeVector(3,4,2);
    lsst::ndarray::Vector<int,3> strides = lsst::ndarray::makeVector(8,2,1);
    lsst::ndarray::Array<double const,3,1> a = lsst::ndarray::external(data,shape,strides);
    lsst::ndarray::Array<double const,3,2> b = lsst::ndarray::static_dimension_cast<2>(a);
    BOOST_CHECK(a == b);
    lsst::ndarray::Array<double,3,1> c = lsst::ndarray::const_array_cast<double>(a);
    BOOST_CHECK(a == c);
    lsst::ndarray::Array<double const,3,3> d = lsst::ndarray::dynamic_dimension_cast<3>(a);
    BOOST_CHECK(a == d);
    lsst::ndarray::Array<double const,3,1> e = d[lsst::ndarray::view()(0,4,2)()];
    lsst::ndarray::Array<double const,3,3> f = lsst::ndarray::dynamic_dimension_cast<3>(e);
    BOOST_CHECK(f.empty());
}

BOOST_AUTO_TEST_CASE(complex) {
    std::complex<double> data[3*4*2] = { std::complex<double>(0.0,0.0) };
    lsst::ndarray::Vector<int,3> shape = lsst::ndarray::makeVector(3,4,2);
    lsst::ndarray::Vector<int,3> strides = lsst::ndarray::makeVector(8,2,1);
    lsst::ndarray::Array<std::complex<double>,3,3> a = lsst::ndarray::external(data,shape,strides);
    lsst::ndarray::Array<double const,3,0> re(getReal(a));
    lsst::ndarray::Array<double const,3,0> im(getImag(a));
    a[1][2][0] = std::complex<double>(4.5,1.2);
    BOOST_CHECK(re[1][2][0] == 4.5);
    BOOST_CHECK(im[1][2][0] == 1.2);
}

BOOST_AUTO_TEST_CASE(indexing) {
    double data[3*4*2] = { 
         0, 1, 2, 3, 4, 5, 6, 7,
         8, 9,10,11,12,13,14,15,
        16,17,18,19,20,21,22,23,
    };
    lsst::ndarray::Vector<int,3> a_shape = lsst::ndarray::makeVector(4,3,2);
    lsst::ndarray::Vector<int,3> a_strides = lsst::ndarray::makeVector(6,2,1);
    lsst::ndarray::Array<double,3,3> a = lsst::ndarray::external(data, a_shape, a_strides);
    BOOST_CHECK(a.front().shallow() == a[0].shallow());
    BOOST_CHECK(a.back().shallow() == a[a_shape[0]-1].shallow());
    int n = 0;
    for (int i=0; i<a_shape[0]; ++i) {
        for (int j=0; j<a_shape[1]; ++j) {
            for (int k=0; k<a_shape[2]; ++k) {
                BOOST_CHECK_EQUAL(a[i][j][k], n);
                ++n;
            }
        }
    }
    lsst::ndarray::Vector<int,2> b_shape = lsst::ndarray::makeVector(8,3);
    lsst::ndarray::Vector<int,2> b_strides = lsst::ndarray::makeVector(1,8);
    lsst::ndarray::Array<double,2> b = lsst::ndarray::external(data, b_shape, b_strides);
    for (int i=0; i<b_shape[0]; ++i) {
        for (int j=0; j<b_shape[1]; ++j) {
            BOOST_CHECK_EQUAL(b[i][j], i+8*j);
        }
    }
    lsst::ndarray::Vector<int,2> c_shape = lsst::ndarray::makeVector(4,3);
    lsst::ndarray::Vector<int,2> c_strides = lsst::ndarray::makeVector(1,8);
    lsst::ndarray::Array<double,2> c = lsst::ndarray::external(data, c_shape, c_strides);
    for (int i=0; i<c_shape[0]; ++i) {
        for (int j=0; j<c_shape[1]; ++j) {
            BOOST_CHECK_EQUAL(c[i][j], i+8*j);
        }
    }
}

BOOST_AUTO_TEST_CASE(iterators) {
    double data[3*4*2] = { 
         0, 1, 2, 3, 4, 5, 6, 7,
         8, 9,10,11,12,13,14,15,
        16,17,18,19,20,21,22,23,
    };
    lsst::ndarray::Vector<int,3> a_shape = lsst::ndarray::makeVector(4,3,2);
    lsst::ndarray::Vector<int,3> a_strides = lsst::ndarray::makeVector(6,2,1);
    lsst::ndarray::Array<double,3,3> a = lsst::ndarray::external(data, a_shape, a_strides);
    lsst::ndarray::Array<double,3,3>::Iterator ai_iter = a.begin();
    lsst::ndarray::Array<double,3,3>::Iterator const ai_end = a.end();
    for (int i=0; ai_iter != ai_end; ++i, ++ai_iter) {
        lsst::ndarray::Array<double,3,3>::Reference::Iterator aj_iter = ai_iter->begin();
        lsst::ndarray::Array<double,3,3>::Reference::Iterator const aj_end = ai_iter->end();
        for (int j=0; aj_iter != aj_end; ++j, ++aj_iter) {
            lsst::ndarray::Array<double,3,3>::Reference::Reference::Iterator ak_iter = aj_iter->begin();
            lsst::ndarray::Array<double,3,3>::Reference::Reference::Iterator const ak_end = aj_iter->end();
            for (int k=0; ak_iter != ak_end; ++k, ++ak_iter) {
                BOOST_CHECK_EQUAL(a[i][j][k], *ak_iter);
            }
        }
    }
    lsst::ndarray::Vector<int,2> b_shape = lsst::ndarray::makeVector(4,3);
    lsst::ndarray::Vector<int,2> b_strides = lsst::ndarray::makeVector(1,8);
    lsst::ndarray::Array<double,2> b = lsst::ndarray::external(data, b_shape, b_strides);
    lsst::ndarray::Array<double,2>::Iterator bi_iter = b.begin();
    lsst::ndarray::Array<double,2>::Iterator const bi_end = b.end();
    for (int i=0; bi_iter != bi_end; ++i, ++bi_iter) {
        lsst::ndarray::Array<double,2>::Reference::Iterator bj_iter = bi_iter->begin();
        lsst::ndarray::Array<double,2>::Reference::Iterator const bj_end = bi_iter->end();
        for (int j=0; bj_iter != bj_end; ++j, ++bj_iter) {
            BOOST_CHECK_EQUAL(b[i][j], *bj_iter);
        }
    }

}

BOOST_AUTO_TEST_CASE(views) {
    lsst::ndarray::Vector<int,3> shape = lsst::ndarray::makeVector(4,3,2);
    lsst::ndarray::Array<double,3,3> a = lsst::ndarray::allocate(shape);
    BOOST_CHECK(a == a[lsst::ndarray::view()()].shallow());
    BOOST_CHECK(a == a[lsst::ndarray::view()].shallow());
    BOOST_CHECK(a[1].shallow() == a[lsst::ndarray::view(1)].shallow());
    BOOST_CHECK(a[1][2].shallow() == a[lsst::ndarray::view(1)(2)].shallow());
    BOOST_CHECK(a != a[lsst::ndarray::view(0,3)].shallow());
    lsst::ndarray::Array<double const,2> b = a[lsst::ndarray::view()(1,3)(0)];
    BOOST_CHECK(b.getShape() == lsst::ndarray::makeVector(4,2));
    BOOST_CHECK(b.getStrides() == lsst::ndarray::makeVector(6,2));
    BOOST_CHECK(b.getData() == a.getData() + 2);
    lsst::ndarray::Array<double const,2> c = b[lsst::ndarray::view(0,4,2)()];
    BOOST_CHECK(c.getShape() == lsst::ndarray::makeVector(2,2));
    BOOST_CHECK(c.getStrides() == lsst::ndarray::makeVector(12,2));
    BOOST_CHECK(c.getData() == b.getData());
}

#ifndef GCC_45

BOOST_AUTO_TEST_CASE(predicates) {
    double data1[3*4*2] = { 
         0, 1, 2, 3, 4, 5, 6, 7,
         8, 9,10,11,12,13,14,15,
        16,17,18,19,20,21,22,23,
    };
    double data2[3*4*2] = { 
         0, 1, 2, 3, 4, 5, 6, 7,
         8, 9,10,11,12,13,14,15,
        16,17,18,19,20,21,22,23,
    };
    lsst::ndarray::Vector<int,3> shape = lsst::ndarray::makeVector(4,3,2);
    lsst::ndarray::Vector<int,3> strides = lsst::ndarray::makeVector(6,2,1);
    lsst::ndarray::Array<double const,3,3> a = lsst::ndarray::external(data1, shape, strides);
    lsst::ndarray::Array<bool,3,2> b = lsst::ndarray::allocate(shape);
    lsst::ndarray::Array<bool,3> c = lsst::ndarray::allocate(shape);
    lsst::ndarray::Array<double,3,1> d = lsst::ndarray::external(data2,shape,strides);
    b.deep() = equal(a, 3.0);
    c.deep() = logical_not(b);
    BOOST_CHECK(a != d);
    BOOST_CHECK(all(equal(a, d)));
    BOOST_CHECK(lsst::ndarray::any(equal(a, d)));
    d[3][1][0] = 5.0;
    BOOST_CHECK(!lsst::ndarray::all(equal(a, d)));
    BOOST_CHECK(lsst::ndarray::any(equal(a, d)));
    BOOST_CHECK(lsst::ndarray::any(not_equal(a, d)));
    d.deep() = -5.0;
    BOOST_CHECK(!lsst::ndarray::all(equal(a, d)));
    BOOST_CHECK(lsst::ndarray::all(not_equal(a, d)));
    BOOST_CHECK(lsst::ndarray::all(greater(a, d)));
    BOOST_CHECK(!lsst::ndarray::any(equal(a, d)));
    for (int i=0; i<shape[0]; ++i) {
        for (int j=0; j<shape[1]; ++j) {
            for (int k=0; k<shape[2]; ++k) {
                if (a[i][j][k] == 3) {
                    BOOST_CHECK_EQUAL(b[i][j][k], true);
                    BOOST_CHECK_EQUAL(c[i][j][k], false);
                } else {
                    BOOST_CHECK_EQUAL(b[i][j][k], false);
                    BOOST_CHECK_EQUAL(c[i][j][k], true);
                }
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(allclose) {
   float data[3*4*2] = { 
         0, 1, 2, 3, 4, 5, 6, 7,
         8, 9,10,11,12,13,14,15,
        16,17,18,19,20,21,22,23,
    };
    lsst::ndarray::Vector<int,3> shape = lsst::ndarray::makeVector(4,3,2);
    lsst::ndarray::Vector<int,3> strides = lsst::ndarray::makeVector(6,2,1);
    lsst::ndarray::Array<float,3,2> a = lsst::ndarray::external(data,shape,strides);
    lsst::ndarray::Array<double,4,4> b = lsst::ndarray::allocate(lsst::ndarray::concatenate(shape, 3));
    lsst::ndarray::Array<double,3,3> c = lsst::ndarray::allocate(shape);
    c.deep() = a + 1.2;
    b.deep() = a + 1.2;
    BOOST_CHECK(lsst::ndarray::allclose(c, b));
    b.deep() += 1E-9;
    BOOST_CHECK(lsst::ndarray::allclose(c, b, 1E-8));
    BOOST_CHECK(!lsst::ndarray::allclose(c, b, 1E-10));
}

BOOST_AUTO_TEST_CASE(binary_ops) {
    float data[3*4*2] = { 
         0, 1, 2, 3, 4, 5, 6, 7,
         8, 9,10,11,12,13,14,15,
        16,17,18,19,20,21,22,23,
    };
    lsst::ndarray::Vector<int,3> shape = lsst::ndarray::makeVector(4,3,2);
    lsst::ndarray::Vector<int,3> strides = lsst::ndarray::makeVector(6,2,1);
    lsst::ndarray::Array<float,3,2> a = lsst::ndarray::external(data,shape,strides);
    lsst::ndarray::Array<double,3,1> b = lsst::ndarray::allocate(shape);
    lsst::ndarray::Array<double,3,3> c = lsst::ndarray::allocate(shape);
    c.deep() = 0.0;
    double q = 1.2;
    b.deep() = a + q;
    c.deep() -= (a * b - q);
    for (int i=0; i<shape[0]; ++i) {
        for (int j=0; j<shape[1]; ++j) {
            for (int k=0; k<shape[2]; ++k) {
                BOOST_CHECK_CLOSE(b[i][j][k],a[i][j][k]+q,1E-8);
                BOOST_CHECK_CLOSE(- (b[i][j][k] * a[i][j][k] - q), c[i][j][k],1E-8);
            }
        }
    }
    BOOST_CHECK(lsst::ndarray::allclose(b, a + q));
}

BOOST_AUTO_TEST_CASE(broadcasting) {
    double data3[3*4*2] = { 
         0, 1, 2, 3, 4, 5, 6, 7,
         8, 9,10,11,12,13,14,15,
        16,17,18,19,20,21,22,23,
    };
    double data2[3*4] = { 
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
    };
    double data32[3*4*2] = { 
        0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11,
    };
    lsst::ndarray::Vector<int,3> shape3 = lsst::ndarray::makeVector(3,4,2);
    lsst::ndarray::Vector<int,3> strides3 = lsst::ndarray::makeVector(8,2,1);
    lsst::ndarray::Vector<int,2> shape2 = lsst::ndarray::makeVector(3,4);
    lsst::ndarray::Vector<int,2> strides2 = lsst::ndarray::makeVector(4,1);
    lsst::ndarray::Array<double const,3,3> a3 = lsst::ndarray::external(data3,shape3,strides3);
    lsst::ndarray::Array<double const,3,3> a32 = lsst::ndarray::external(data32,shape3,strides3);
    lsst::ndarray::Array<double const,2,2> a2 = lsst::ndarray::external(data2,shape2,strides2);
    lsst::ndarray::Array<double,3,3> b3 = lsst::ndarray::copy(a3);
    lsst::ndarray::Array<double,3,3> b32 = lsst::ndarray::copy(a32);
    lsst::ndarray::Array<double,2,2> b2 = lsst::ndarray::copy(a2);
    lsst::ndarray::Array<double,3,3> c3 = lsst::ndarray::allocate(shape3);
    lsst::ndarray::Array<double,3,3> c32 = lsst::ndarray::allocate(shape3);
    c3.deep() = 0.0;
    c3.deep() -= 2.0 * b2;
    c32.deep() = 0.0;
    c32.deep() -= 2.0 * b32;
    BOOST_CHECK(all(equal(c3, c32)));
    c3.deep() += b2;
    c32.deep() += b32;
    BOOST_CHECK(all(equal(c3, c32)));
}

#endif

BOOST_AUTO_TEST_CASE(assignment) {
    double data[3*4*2] = { 
         0, 1, 2, 3, 4, 5, 6, 7,
         8, 9,10,11,12,13,14,15,
        16,17,18,19,20,21,22,23,
    };
    lsst::ndarray::Vector<int,3> shape = lsst::ndarray::makeVector(3,4,2);
    lsst::ndarray::Vector<int,3> strides = lsst::ndarray::makeVector(8,2,1);
    lsst::ndarray::Array<double,3,3> a = lsst::ndarray::external(data,shape,strides);
    lsst::ndarray::Array<double,3,3> b = lsst::ndarray::allocate(shape);
    b.deep() = a;
    lsst::ndarray::Array<double,3,2> c = lsst::ndarray::allocate(shape);
    lsst::ndarray::Array<double const,3,1> d = c;
    c.deep() = a;
    BOOST_CHECK(a.shallow() != b.shallow());
    BOOST_CHECK(a.shallow() != c.shallow());
    int n = 0;
    for (int i=0; i<shape[0]; ++i) {
        for (int j=0; j<shape[1]; ++j) {
            for (int k=0; k<shape[2]; ++k) {
                BOOST_CHECK_EQUAL(b[i][j][k],n);
                BOOST_CHECK_EQUAL(c[i][j][k],n);
                BOOST_CHECK_EQUAL(d[i][j][k],n);
                ++n;
            }
        }
    }
    double q = 5.3;
    double p = 4.2;
    b.deep() = q;
    c.deep() = p;
    for (int i=0; i<shape[0]; ++i) {
        for (int j=0; j<shape[1]; ++j) {
            for (int k=0; k<shape[2]; ++k) {
                BOOST_CHECK_EQUAL(b[i][j][k],q);
                BOOST_CHECK_EQUAL(c[i][j][k],p);
            }
        }
    }
    b.deep() += a;
    c.deep() -= a;
    for (int i=0; i<shape[0]; ++i) {
        for (int j=0; j<shape[1]; ++j) {
            for (int k=0; k<shape[2]; ++k) {
                BOOST_CHECK_CLOSE(b[i][j][k],q,1E-8);
                BOOST_CHECK_CLOSE(c[i][j][k],p,1E-8);
                ++q;
                --p;
            }
        }
    }
}


BOOST_AUTO_TEST_CASE(transpose) {
    double data[3*4*2] = { 
         0, 1, 2, 3, 4, 5, 6, 7,
         8, 9,10,11,12,13,14,15,
        16,17,18,19,20,21,22,23,
    };
    lsst::ndarray::Vector<int,3> shape = lsst::ndarray::makeVector(3,4,2);
    lsst::ndarray::Vector<int,3> strides = lsst::ndarray::makeVector(8,2,1);
    lsst::ndarray::Array<double,3,3> a = lsst::ndarray::external(data,shape,strides);
    lsst::ndarray::Array<double const,3> b = a.transpose();
    lsst::ndarray::Array<double const,3> c = a.transpose(lsst::ndarray::makeVector(1,0,2));
    for (int i=0; i<shape[0]; ++i) {
        for (int j=0; j<shape[1]; ++j) {
            for (int k=0; k<shape[2]; ++k) {
                BOOST_CHECK_EQUAL(a[i][j][k],b[k][j][i]);
                BOOST_CHECK_EQUAL(a[i][j][k],c[j][i][k]);
            }
        }
    }
    BOOST_CHECK(a[lsst::ndarray::view()(1)(1)].shallow() == b[lsst::ndarray::view(1)(1)()].shallow());
    BOOST_CHECK(a[lsst::ndarray::view(0)()(1)].shallow() == b[lsst::ndarray::view(1)()(0)].shallow());
    BOOST_CHECK(a[lsst::ndarray::view(0)(0)()].shallow() == b[lsst::ndarray::view()(0)(0)].shallow());
    BOOST_CHECK(a[lsst::ndarray::view()(1)(1)].shallow() == c[lsst::ndarray::view(1)()(1)].shallow());
    BOOST_CHECK(a[lsst::ndarray::view(0)()(1)].shallow() == c[lsst::ndarray::view()(0)(1)].shallow());
    BOOST_CHECK(a[lsst::ndarray::view(0)(0)()].shallow() == c[lsst::ndarray::view(0)(0)()].shallow());
}

BOOST_AUTO_TEST_CASE(flatten) {
    double data[3*4*2] = { 
         0, 1, 2, 3, 4, 5, 6, 7,
         8, 9,10,11,12,13,14,15,
        16,17,18,19,20,21,22,23,
    };
    lsst::ndarray::Vector<int,3> a_shape = lsst::ndarray::makeVector(4,3,2);
    lsst::ndarray::Vector<int,3> a_strides = lsst::ndarray::makeVector(6,2,1);
    lsst::ndarray::Array<double,3,3> a = lsst::ndarray::external(data,a_shape,a_strides);
    lsst::ndarray::Array<double,2,2> b = lsst::ndarray::flatten<2>(a);
    lsst::ndarray::Array<double,2,2> b_check = lsst::ndarray::external(
        data, lsst::ndarray::makeVector(4,6), lsst::ndarray::makeVector(6,1)
    );
    BOOST_CHECK(b.shallow() == b_check.shallow());
    lsst::ndarray::Array<double,1,1> c = lsst::ndarray::flatten<1>(a);
    lsst::ndarray::Array<double,1,1> c_check = lsst::ndarray::external(
        data, lsst::ndarray::makeVector(24), lsst::ndarray::makeVector(1)
    );
}

BOOST_AUTO_TEST_CASE(unique) {
    lsst::ndarray::Array<double,2,2> a = lsst::ndarray::allocate(5,4);
    BOOST_CHECK(a.isUnique());
    lsst::ndarray::Array<double const,2,2> b(a);
    BOOST_CHECK(!a.isUnique());
    BOOST_CHECK(!b.isUnique());
    a = lsst::ndarray::Array<double,2,2>();
    BOOST_CHECK(b.isUnique());    
    lsst::ndarray::Array<double const,2,1> c = b[lsst::ndarray::view(1,4)(1,3)];
    BOOST_CHECK(!c.isUnique());
    BOOST_CHECK(!b.isUnique());
    b = lsst::ndarray::Array<double const,2,2>();
    BOOST_CHECK(c.isUnique());
    lsst::ndarray::Array<double const,2,1> d = lsst::ndarray::allocate(6,3);
    d.swap(c);
    BOOST_CHECK(c.isUnique());
    BOOST_CHECK(d.isUnique());
}

BOOST_AUTO_TEST_CASE(ticket1720) {
    lsst::ndarray::Array<int,2,2> a = lsst::ndarray::allocate(5,4);
    a.deep() = 0;
    lsst::ndarray::Array<int,2,2> b = lsst::ndarray::allocate(5,4);
    b.deep() = 1;
    a[0] = b[1];
    BOOST_CHECK_EQUAL(a[0][0], 1);
    lsst::ndarray::Array<int,2,0> c(b);
    a[1] = c[1];
    BOOST_CHECK_EQUAL(a[1][0], 1);
}

BOOST_AUTO_TEST_CASE(zeroSize) {
    lsst::ndarray::Array<double,1,1> a = lsst::ndarray::allocate(0);
    BOOST_CHECK(!a.getData());
    BOOST_CHECK_EQUAL(a.getSize<0>(), 0);
    lsst::ndarray::Array<double,2,2> b = lsst::ndarray::allocate(0, 5);
    BOOST_CHECK(!b.getData());
    BOOST_CHECK_EQUAL(b.getSize<0>(), 0);
    BOOST_CHECK_EQUAL(b.getSize<1>(), 5);
    lsst::ndarray::Array<double,2,2> c = lsst::ndarray::allocate(5, 0);
    BOOST_CHECK(!c.getData());
    BOOST_CHECK_EQUAL(c.getSize<0>(), 5);
    BOOST_CHECK_EQUAL(c.getSize<1>(), 0);
}
