// -*- c++ -*-
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
#include "ndarray.h"

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE ndarray
#include "boost/test/unit_test.hpp"

BOOST_AUTO_TEST_CASE(sizes) {
    std::cerr << "sizeof(int): " << sizeof(int) << "\n";
    std::cerr << "sizeof(double*): " << sizeof(double*) << "\n";
    std::cerr << "sizeof(shared_ptr<double>): " << sizeof(boost::shared_ptr<double>) << "\n";
    std::cerr << "sizeof(Core<double,1>): " << sizeof(ndarray::detail::Core<1>) << "\n";
    std::cerr << "sizeof(Core<double,2>): " << sizeof(ndarray::detail::Core<2>) << "\n";
    std::cerr << "sizeof(Core<double,3>): " << sizeof(ndarray::detail::Core<3>) << "\n";
}

BOOST_AUTO_TEST_CASE(vectors) {
    ndarray::Vector<int,3> a = ndarray::makeVector(5,6,7);
    BOOST_CHECK_EQUAL(a[0],5);
    BOOST_CHECK_EQUAL(a[1],6);
    BOOST_CHECK_EQUAL(a[2],7);
    ndarray::Vector<int,3> b(a);
    BOOST_CHECK_EQUAL(a,b);
    ndarray::Vector<double,3> c(a);
    BOOST_CHECK_EQUAL(c[0],5.0);
    BOOST_CHECK_EQUAL(c[1],6.0);
    BOOST_CHECK_EQUAL(c[2],7.0);
    ndarray::Vector<double,3> d(5.0);
    BOOST_CHECK_EQUAL(d[0],5.0);
    BOOST_CHECK_EQUAL(d[1],5.0);
    BOOST_CHECK_EQUAL(d[2],5.0);

    ndarray::Vector<int,0> e;
    ndarray::Vector<int,3> f = ndarray::concatenate(a, e);
    BOOST_CHECK_EQUAL(a, f);

    e = a.last<0>();

    // make sure we can default-construct whether or not T is a number
    ndarray::Vector<boost::shared_ptr<int>,2> g;
    BOOST_CHECK_EQUAL(g, ndarray::makeVector(boost::shared_ptr<int>(), boost::shared_ptr<int>()));
    ndarray::Vector<int,3> h;
    BOOST_CHECK_EQUAL(h, ndarray::makeVector(0, 0, 0));
}

BOOST_AUTO_TEST_CASE(cores) {
    typedef ndarray::detail::Core<3> Core;
    ndarray::Vector<int,3> shape = ndarray::makeVector(4,3,2);
    ndarray::Vector<int,3> strides = ndarray::makeVector(6,2,1);
    Core::Ptr core = Core::create(shape, strides);
    BOOST_CHECK_EQUAL(core->getRC(),1);
    Core::Ptr copy = core;
    BOOST_CHECK_EQUAL(core->getRC(),2);
    copy.reset();
    BOOST_CHECK_EQUAL(core->getRC(),1);
}

BOOST_AUTO_TEST_CASE(allocation) {
    ndarray::Vector<int,3> shape = ndarray::makeVector(5,6,7);
    ndarray::Array<float,3,3> a = ndarray::allocate(shape);
    BOOST_CHECK_EQUAL(a.getShape(), shape);

    ndarray::Array<float,3> b = ndarray::allocate(shape);
    BOOST_CHECK_EQUAL(b.getShape(), shape);
    BOOST_CHECK_EQUAL(b.getSize<0>(), shape[0]);
    BOOST_CHECK_EQUAL(b.getSize<1>(), shape[1]);
    BOOST_CHECK_EQUAL(b.getSize<2>(), shape[2]);
    BOOST_CHECK_EQUAL(b.getStride<0>(), 6*7);
    BOOST_CHECK_EQUAL(b.getStride<1>(), 7);
    BOOST_CHECK_EQUAL(b.getStride<2>(), 1);
    BOOST_CHECK_EQUAL(b.getStrides(), ndarray::makeVector(6*7 ,7, 1));

    ndarray::Array<float,3,-3> c = ndarray::allocate(5,6,7);
    BOOST_CHECK_EQUAL(c.getShape(), shape);
    BOOST_CHECK_EQUAL(c.getSize<0>(), shape[0]);
    BOOST_CHECK_EQUAL(c.getSize<1>(), shape[1]);
    BOOST_CHECK_EQUAL(c.getSize<2>(), shape[2]);
    BOOST_CHECK_EQUAL(c.getStride<0>(), 1);
    BOOST_CHECK_EQUAL(c.getStride<1>(), 5);
    BOOST_CHECK_EQUAL(c.getStride<2>(), 5*6);
    BOOST_CHECK_EQUAL(c.getStrides(), ndarray::makeVector(1, 5, 5*6));
    
}

BOOST_AUTO_TEST_CASE(external) {
    double data[3*4*2] = {0};
    ndarray::Vector<int,3> shape = ndarray::makeVector(3,4,2);
    ndarray::Vector<int,3> strides = ndarray::makeVector(8,2,1);
    ndarray::Array<double,3,3> a = ndarray::external(data,shape,strides);
    BOOST_CHECK_EQUAL(a.getData(), data);
    BOOST_CHECK_EQUAL(a.getShape(), shape);
    BOOST_CHECK_EQUAL(a.getStrides(), strides);
}

BOOST_AUTO_TEST_CASE(conversion) {
    double data[3*4*2] = {0};
    ndarray::Vector<int,3> shape = ndarray::makeVector(3,4,2);
    ndarray::Vector<int,3> strides = ndarray::makeVector(8,2,1);
    ndarray::Array<double,3,3> a = ndarray::external(data,shape,strides);
    ndarray::Array<double const,3> b = a;
}

BOOST_AUTO_TEST_CASE(shallow) {
    double data[3*4*2] = {0};
    ndarray::Vector<int,3> shape = ndarray::makeVector(3,4,2);
    ndarray::Vector<int,3> strides = ndarray::makeVector(8,2,1);
    ndarray::Array<double,3,3> a = ndarray::external(data,shape,strides);
    ndarray::Array<double,3,1> b = ndarray::external(data,shape,strides);
    BOOST_CHECK(a == b);
    BOOST_CHECK(a[2].shallow() == b[2].shallow());
    BOOST_CHECK(a[0][1].shallow() == b[0][1].shallow());
    BOOST_CHECK(a[0][1].shallow() != b[1][2].shallow());
    ndarray::Array<double,3,3> c;
    c = a;
    BOOST_CHECK_EQUAL(a.getData(), c.getData());
    BOOST_CHECK_EQUAL(a.getShape(), c.getShape());
    BOOST_CHECK_EQUAL(a.getStrides(), c.getStrides());
    BOOST_CHECK(a.shallow() == c.shallow());
    ndarray::Array<double,2> d = c[1];
    BOOST_CHECK_EQUAL(d.getData(), c[1].getData());
    BOOST_CHECK_EQUAL(d.getShape(), c[1].getShape());
    BOOST_CHECK_EQUAL(d.getStrides(), c[1].getStrides());
    BOOST_CHECK(d.shallow() == c[1].shallow());
}

BOOST_AUTO_TEST_CASE(casts) {
    double data[3*4*2] = {0};
    ndarray::Vector<int,3> shape = ndarray::makeVector(3,4,2);
    ndarray::Vector<int,3> strides = ndarray::makeVector(8,2,1);
    ndarray::Array<double const,3,1> a = ndarray::external(data,shape,strides);
    ndarray::Array<double const,3,2> b = ndarray::static_dimension_cast<2>(a);
    BOOST_CHECK(a == b);
    ndarray::Array<double,3,1> c = ndarray::const_array_cast<double>(a);
    BOOST_CHECK(a == c);
    ndarray::Array<double const,3,3> d = ndarray::dynamic_dimension_cast<3>(a);
    BOOST_CHECK(a == d);
    ndarray::Array<double const,3,1> e = d[ndarray::view()(0,4,2)()];
    ndarray::Array<double const,3,3> f = ndarray::dynamic_dimension_cast<3>(e);
    BOOST_CHECK(f.empty());
}

BOOST_AUTO_TEST_CASE(complex) {
    std::complex<double> data[3*4*2] = { std::complex<double>(0.0,0.0) };
    ndarray::Vector<int,3> shape = ndarray::makeVector(3,4,2);
    ndarray::Vector<int,3> strides = ndarray::makeVector(8,2,1);
    ndarray::Array<std::complex<double>,3,3> a = ndarray::external(data,shape,strides);
    ndarray::Array<double const,3,0> re(getReal(a));
    ndarray::Array<double const,3,0> im(getImag(a));
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
    ndarray::Vector<int,3> a_shape = ndarray::makeVector(4,3,2);
    ndarray::Vector<int,3> a_strides = ndarray::makeVector(6,2,1);
    ndarray::Array<double,3,3> a = ndarray::external(data, a_shape, a_strides);
    BOOST_CHECK(a.front().shallow() == a[0].shallow());
    BOOST_CHECK(a.back().shallow() == a[a_shape[0]-1].shallow());
    int n = 0;
    for (int i=0; i<a_shape[0]; ++i) {
        for (int j=0; j<a_shape[1]; ++j) {
            for (int k=0; k<a_shape[2]; ++k) {
                BOOST_CHECK_EQUAL(a[i][j][k], n);
                BOOST_CHECK_EQUAL(a(i,j,k), n);
                ++n;
            }
        }
    }
    ndarray::Vector<int,2> b_shape = ndarray::makeVector(8,3);
    ndarray::Vector<int,2> b_strides = ndarray::makeVector(1,8);
    ndarray::Array<double,2> b = ndarray::external(data, b_shape, b_strides);
    for (int i=0; i<b_shape[0]; ++i) {
        for (int j=0; j<b_shape[1]; ++j) {
            BOOST_CHECK_EQUAL(b[i][j], i+8*j);
            BOOST_CHECK_EQUAL(b(i,j), i+8*j);
        }
    }
    ndarray::Vector<int,2> c_shape = ndarray::makeVector(4,3);
    ndarray::Vector<int,2> c_strides = ndarray::makeVector(1,8);
    ndarray::Array<double,2> c = ndarray::external(data, c_shape, c_strides);
    for (int i=0; i<c_shape[0]; ++i) {
        for (int j=0; j<c_shape[1]; ++j) {
            BOOST_CHECK_EQUAL(c[i][j], i+8*j);
            BOOST_CHECK_EQUAL(c(i,j), i+8*j);
        }
    }
}

BOOST_AUTO_TEST_CASE(iterators) {
    double data[3*4*2] = { 
         0, 1, 2, 3, 4, 5, 6, 7,
         8, 9,10,11,12,13,14,15,
        16,17,18,19,20,21,22,23,
    };
    ndarray::Vector<int,3> a_shape = ndarray::makeVector(4,3,2);
    ndarray::Vector<int,3> a_strides = ndarray::makeVector(6,2,1);
    ndarray::Array<double,3,3> a = ndarray::external(data, a_shape, a_strides);
    ndarray::Array<double,3,3>::Iterator ai_iter = a.begin();
    ndarray::Array<double,3,3>::Iterator const ai_end = a.end();
    for (int i=0; ai_iter != ai_end; ++i, ++ai_iter) {
        ndarray::Array<double,3,3>::Reference::Iterator aj_iter = ai_iter->begin();
        ndarray::Array<double,3,3>::Reference::Iterator const aj_end = ai_iter->end();
        for (int j=0; aj_iter != aj_end; ++j, ++aj_iter) {
            ndarray::Array<double,3,3>::Reference::Reference::Iterator ak_iter = aj_iter->begin();
            ndarray::Array<double,3,3>::Reference::Reference::Iterator const ak_end = aj_iter->end();
            for (int k=0; ak_iter != ak_end; ++k, ++ak_iter) {
                BOOST_CHECK_EQUAL(a[i][j][k], *ak_iter);
            }
        }
    }
    ndarray::Vector<int,2> b_shape = ndarray::makeVector(4,3);
    ndarray::Vector<int,2> b_strides = ndarray::makeVector(1,8);
    ndarray::Array<double,2> b = ndarray::external(data, b_shape, b_strides);
    ndarray::Array<double,2>::Iterator bi_iter = b.begin();
    ndarray::Array<double,2>::Iterator const bi_end = b.end();
    for (int i=0; bi_iter != bi_end; ++i, ++bi_iter) {
        ndarray::Array<double,2>::Reference::Iterator bj_iter = bi_iter->begin();
        ndarray::Array<double,2>::Reference::Iterator const bj_end = bi_iter->end();
        for (int j=0; bj_iter != bj_end; ++j, ++bj_iter) {
            BOOST_CHECK_EQUAL(b[i][j], *bj_iter);
        }
    }

}

BOOST_AUTO_TEST_CASE(views) {
    ndarray::Vector<int,3> shape = ndarray::makeVector(4,3,2);
    ndarray::Array<double,3,3> a = ndarray::allocate(shape);
    BOOST_CHECK(a == a[ndarray::view()()].shallow());
    BOOST_CHECK(a == a[ndarray::view()].shallow());
    BOOST_CHECK(a[1].shallow() == a[ndarray::view(1)].shallow());
    BOOST_CHECK(a[1][2].shallow() == a[ndarray::view(1)(2)].shallow());
    BOOST_CHECK(a != a[ndarray::view(0,3)].shallow());
    ndarray::Array<double const,2> b = a[ndarray::view()(1,3)(0)];
    BOOST_CHECK(b.getShape() == ndarray::makeVector(4,2));
    BOOST_CHECK(b.getStrides() == ndarray::makeVector(6,2));
    BOOST_CHECK(b.getData() == a.getData() + 2);
    ndarray::Array<double const,2> c = b[ndarray::view(0,4,2)()];
    BOOST_CHECK(c.getShape() == ndarray::makeVector(2,2));
    BOOST_CHECK(c.getStrides() == ndarray::makeVector(12,2));
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
    ndarray::Vector<int,3> shape = ndarray::makeVector(4,3,2);
    ndarray::Vector<int,3> strides = ndarray::makeVector(6,2,1);
    ndarray::Array<double const,3,3> a = ndarray::external(data1, shape, strides);
    ndarray::Array<bool,3,2> b = ndarray::allocate(shape);
    ndarray::Array<bool,3> c = ndarray::allocate(shape);
    ndarray::Array<double,3,1> d = ndarray::external(data2,shape,strides);
    b.deep() = equal(a, 3.0);
    c.deep() = logical_not(b);
    BOOST_CHECK(a != d);
    BOOST_CHECK(all(equal(a, d)));
    BOOST_CHECK(ndarray::any(equal(a, d)));
    d[3][1][0] = 5.0;
    BOOST_CHECK(!ndarray::all(equal(a, d)));
    BOOST_CHECK(ndarray::any(equal(a, d)));
    BOOST_CHECK(ndarray::any(not_equal(a, d)));
    d.deep() = -5.0;
    BOOST_CHECK(!ndarray::all(equal(a, d)));
    BOOST_CHECK(ndarray::all(not_equal(a, d)));
    BOOST_CHECK(ndarray::all(greater(a, d)));
    BOOST_CHECK(!ndarray::any(equal(a, d)));
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
    ndarray::Vector<int,3> shape = ndarray::makeVector(4,3,2);
    ndarray::Vector<int,3> strides = ndarray::makeVector(6,2,1);
    ndarray::Array<float,3,2> a = ndarray::external(data,shape,strides);
    ndarray::Array<double,4,4> b = ndarray::allocate(ndarray::concatenate(shape, 3));
    ndarray::Array<double,3,3> c = ndarray::allocate(shape);
    c.deep() = a + 1.2;
    b.deep() = a + 1.2;
    BOOST_CHECK(ndarray::allclose(c, b));
    b.deep() += 1E-9;
    BOOST_CHECK(ndarray::allclose(c, b, 1E-8));
    BOOST_CHECK(!ndarray::allclose(c, b, 1E-10));
}

BOOST_AUTO_TEST_CASE(binary_ops) {
    float data[3*4*2] = { 
         0, 1, 2, 3, 4, 5, 6, 7,
         8, 9,10,11,12,13,14,15,
        16,17,18,19,20,21,22,23,
    };
    ndarray::Vector<int,3> shape = ndarray::makeVector(4,3,2);
    ndarray::Vector<int,3> strides = ndarray::makeVector(6,2,1);
    ndarray::Array<float,3,2> a = ndarray::external(data,shape,strides);
    ndarray::Array<double,3,1> b = ndarray::allocate(shape);
    ndarray::Array<double,3,3> c = ndarray::allocate(shape);
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
    BOOST_CHECK(ndarray::allclose(b, a + q));
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
    ndarray::Vector<int,3> shape3 = ndarray::makeVector(3,4,2);
    ndarray::Vector<int,3> strides3 = ndarray::makeVector(8,2,1);
    ndarray::Vector<int,2> shape2 = ndarray::makeVector(3,4);
    ndarray::Vector<int,2> strides2 = ndarray::makeVector(4,1);
    ndarray::Array<double const,3,3> a3 = ndarray::external(data3,shape3,strides3);
    ndarray::Array<double const,3,3> a32 = ndarray::external(data32,shape3,strides3);
    ndarray::Array<double const,2,2> a2 = ndarray::external(data2,shape2,strides2);
    ndarray::Array<double,3,3> b3 = ndarray::copy(a3);
    ndarray::Array<double,3,3> b32 = ndarray::copy(a32);
    ndarray::Array<double,2,2> b2 = ndarray::copy(a2);
    ndarray::Array<double,3,3> c3 = ndarray::allocate(shape3);
    ndarray::Array<double,3,3> c32 = ndarray::allocate(shape3);
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
    ndarray::Vector<int,3> shape = ndarray::makeVector(3,4,2);
    ndarray::Vector<int,3> strides = ndarray::makeVector(8,2,1);
    ndarray::Array<double,3,3> a = ndarray::external(data,shape,strides);
    ndarray::Array<double,3,3> b = ndarray::allocate(shape);
    b.deep() = a;
    ndarray::Array<double,3,2> c = ndarray::allocate(shape);
    ndarray::Array<double const,3,1> d = c;
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
    ndarray::Vector<int,3> shape = ndarray::makeVector(3,4,2);
    ndarray::Vector<int,3> strides = ndarray::makeVector(8,2,1);
    ndarray::Array<double,3,3> a = ndarray::external(data,shape,strides);
    ndarray::Array<double const,3,-3> b = a.transpose();
    ndarray::Array<double const,3> c = a.transpose(ndarray::makeVector(1,0,2));
    for (int i=0; i<shape[0]; ++i) {
        for (int j=0; j<shape[1]; ++j) {
            for (int k=0; k<shape[2]; ++k) {
                BOOST_CHECK_EQUAL(a[i][j][k],b[k][j][i]);
                BOOST_CHECK_EQUAL(a[i][j][k],c[j][i][k]);
            }
        }
    }
    BOOST_CHECK(a[ndarray::view()(1)(1)].shallow() == b[ndarray::view(1)(1)()].shallow());
    BOOST_CHECK(a[ndarray::view(0)()(1)].shallow() == b[ndarray::view(1)()(0)].shallow());
    BOOST_CHECK(a[ndarray::view(0)(0)()].shallow() == b[ndarray::view()(0)(0)].shallow());
    BOOST_CHECK(a[ndarray::view()(1)(1)].shallow() == c[ndarray::view(1)()(1)].shallow());
    BOOST_CHECK(a[ndarray::view(0)()(1)].shallow() == c[ndarray::view()(0)(1)].shallow());
    BOOST_CHECK(a[ndarray::view(0)(0)()].shallow() == c[ndarray::view(0)(0)()].shallow());

    {
        ndarray::Array<double const,2,2> a1 = a[ndarray::view(0)()()];
        ndarray::Array<double const,2,-2> b1 = b[ndarray::view()()(0)];
        BOOST_CHECK(b1.transpose().shallow() == a1.shallow());
        BOOST_CHECK(a1.transpose().shallow() == b1.shallow());
    }
    {
        ndarray::Array<double const,3,3> a1 = a[ndarray::view(1,3)()()];
        ndarray::Array<double const,3,-3> b1 = b[ndarray::view()()(1,3)];
        BOOST_CHECK(b1.transpose().shallow() == a1.shallow());
        BOOST_CHECK(a1.transpose().shallow() == b1.shallow());
    }
    {
        ndarray::Array<double const,3,2> a1 = a[ndarray::view(0,3,2)()()];
        ndarray::Array<double const,3,-2> b1 = b[ndarray::view()()(0,3,2)];
        BOOST_CHECK(b1.transpose().shallow() == a1.shallow());
        BOOST_CHECK(a1.transpose().shallow() == b1.shallow());
    }
    {
        ndarray::Array<double const,3,2> a1 = a[ndarray::view()(1,3)()];
        ndarray::Array<double const,3,-2> b1 = b[ndarray::view()(1,3)()];
        BOOST_CHECK(b1.transpose().shallow() == a1.shallow());
        BOOST_CHECK(a1.transpose().shallow() == b1.shallow());
    }
    {
        ndarray::Array<double const,3,1> a1 = a[ndarray::view()(0,4,2)()];
        ndarray::Array<double const,3,-1> b1 = b[ndarray::view()(0,4,2)()];
        BOOST_CHECK(b1.transpose().shallow() == a1.shallow());
        BOOST_CHECK(a1.transpose().shallow() == b1.shallow());
    }
    {
        ndarray::Array<double const,1,1> a1 = a[0][0];
        ndarray::Array<double const,1,-1> b1 = a1;
        ndarray::Array<double const,1,1> c1 = b1;
    }
}

BOOST_AUTO_TEST_CASE(flatten) {
    double data[3*4*2] = { 
         0, 1, 2, 3, 4, 5, 6, 7,
         8, 9,10,11,12,13,14,15,
        16,17,18,19,20,21,22,23,
    };
    ndarray::Vector<int,3> a_shape = ndarray::makeVector(4,3,2);
    ndarray::Vector<int,3> a_strides = ndarray::makeVector(6,2,1);
    ndarray::Array<double,3,3> a = ndarray::external(data,a_shape,a_strides);
    ndarray::Array<double,2,2> b = ndarray::flatten<2>(a);
    ndarray::Array<double,2,2> b_check = ndarray::external(
        data, ndarray::makeVector(4,6), ndarray::makeVector(6,1)
    );
    BOOST_CHECK(b.shallow() == b_check.shallow());
    ndarray::Array<double,1,1> c = ndarray::flatten<1>(a);
    ndarray::Array<double,1,1> c_check = ndarray::external(
        data, ndarray::makeVector(24), ndarray::makeVector(1)
    );
}

BOOST_AUTO_TEST_CASE(unique) {
    ndarray::Array<double,2,2> a = ndarray::allocate(5,4);
    BOOST_CHECK(a.isUnique());
    ndarray::Array<double const,2,2> b(a);
    BOOST_CHECK(!a.isUnique());
    BOOST_CHECK(!b.isUnique());
    a = ndarray::Array<double,2,2>();
    BOOST_CHECK(b.isUnique());    
    ndarray::Array<double const,2,1> c = b[ndarray::view(1,4)(1,3)];
    BOOST_CHECK(!c.isUnique());
    BOOST_CHECK(!b.isUnique());
    b = ndarray::Array<double const,2,2>();
    BOOST_CHECK(c.isUnique());
    ndarray::Array<double const,2,1> d = ndarray::allocate(6,3);
    d.swap(c);
    BOOST_CHECK(c.isUnique());
    BOOST_CHECK(d.isUnique());
}

BOOST_AUTO_TEST_CASE(ticket1720) {
    ndarray::Array<int,2,2> a = ndarray::allocate(5,4);
    a.deep() = 0;
    ndarray::Array<int,2,2> b = ndarray::allocate(5,4);
    b.deep() = 1;
    a[0] = b[1];
    BOOST_CHECK_EQUAL(a[0][0], 1);
    ndarray::Array<int,2,0> c(b);
    a[1] = c[1];
    BOOST_CHECK_EQUAL(a[1][0], 1);
}

BOOST_AUTO_TEST_CASE(zeroSize) {
    ndarray::Array<double,1,1> a = ndarray::allocate(0);
    BOOST_CHECK(!a.getData());
    BOOST_CHECK_EQUAL(a.getSize<0>(), 0);
    ndarray::Array<double,2,2> b = ndarray::allocate(0, 5);
    BOOST_CHECK(!b.getData());
    BOOST_CHECK_EQUAL(b.getSize<0>(), 0);
    BOOST_CHECK_EQUAL(b.getSize<1>(), 5);
    ndarray::Array<double,2,2> c = ndarray::allocate(5, 0);
    BOOST_CHECK(!c.getData());
    BOOST_CHECK_EQUAL(c.getSize<0>(), 5);
    BOOST_CHECK_EQUAL(c.getSize<1>(), 0);
}

BOOST_AUTO_TEST_CASE(manager) {
    ndarray::Array<double,1,1> a = ndarray::allocate(5);
    ndarray::Array<double,1,1> b 
        = ndarray::external(a.getData(), a.getShape(), a.getStrides(), a.getManager());
    BOOST_CHECK_EQUAL(a.getManager(), b.getManager()); // no extra indirection in makeManager
}
