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
#include <ndarray/fft.h>

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE ndarray-fft
#include "boost/test/unit_test.hpp"

#include <sstream>

#ifndef GCC_45

template <typename Derived1, typename Derived2>
static boost::test_tools::predicate_result
compareRelative(
    ndarray::ExpressionBase<Derived1> const & a, 
    ndarray::ExpressionBase<Derived2> const & b,
    double tolerance = 1E-4
) {
    ndarray::ApproximatelyEqual<typename Derived1::Element, typename Derived2::Element> predicate(tolerance);
    if (ndarray::all(ndarray::vectorize(predicate,a,b))) return true;
    boost::test_tools::predicate_result r(false);
    std::ostringstream oss;
    oss << "\n" << a << "\n != \n" << b << "\n";
    r.message() << oss.str();
    return r;
};

template <typename Derived1, typename Derived2>
static boost::test_tools::predicate_result
compareAbsolute(
    ndarray::ExpressionBase<Derived1> const & a,
    ndarray::ExpressionBase<Derived2> const & b,
    double tolerance = 1E-4
) {
    if (ndarray::all(less(a-b, tolerance)) && ndarray::all(less(b-a, tolerance))) return true;
    boost::test_tools::predicate_result r(false);
    std::ostringstream oss;
    oss << "\n" << a << "\n != \n" << b << "\n";
    r.message() << oss.str();
    return r;
};

template <typename T, int N>
struct FourierTransformTester {
    typedef ndarray::FourierTransform<T,N> FFT;

    static void testSingle(
        typename FFT::Index const & shape, 
        typename FFT::ElementX const * xData, 
        typename FFT::ElementK const * kData
    ) {
        typename FFT::ArrayX xIn = FFT::initializeX(shape);
        typename FFT::ArrayK kIn = FFT::initializeK(shape);
        std::copy(xData,xData+xIn.getNumElements(),xIn.getData());
        std::copy(kData,kData+kIn.getNumElements(),kIn.getData());
        typename FFT::ArrayX x;
        typename FFT::ArrayK k;
        typename FFT::Ptr forward = FFT::planForward(shape,x,k);
        typename FFT::Ptr inverse = FFT::planInverse(shape,k,x);
        x.deep() = xIn;
        forward->execute();
        BOOST_CHECK(compareRelative(k,kIn));
        x.deep() = 0.0;
        inverse->execute();
        x.deep() /= shape.product();
        BOOST_CHECK(compareRelative(x,xIn));
    }

    static void testMultiplex(
        typename FFT::MultiplexIndex const & shape, 
        typename FFT::ElementX const * xData, 
        typename FFT::ElementK const * kData
    ) {
        typename FFT::MultiplexArrayX xIn = FFT::initializeX(shape);
        typename FFT::MultiplexArrayK kIn = FFT::initializeK(shape);
        std::copy(xData,xData+xIn.getNumElements(),xIn.getData());
        std::copy(kData,kData+kIn.getNumElements(),kIn.getData());
        typename FFT::MultiplexArrayX x;
        typename FFT::MultiplexArrayK k;
        typename FFT::Ptr forward = FFT::planMultiplexForward(shape,x,k);
        typename FFT::Ptr inverse = FFT::planMultiplexInverse(shape,k,x);
        x.deep() = xIn;
        forward->execute();
        BOOST_CHECK(compareRelative(k,kIn));
        x.deep() = 0.0;
        inverse->execute();
        x.deep() /= (shape.product() / shape[0]);
        BOOST_CHECK(compareRelative(x,xIn));
    }
};

BOOST_AUTO_TEST_CASE(real_1d) {
    double xData1[] = { -0.28131077, -0.25505012, -0.35444799,  1.77553825,  1.655009 };
    std::complex<double> kData1[] = {
        std::complex<double>(2.53973837,0.0), 
        std::complex<double>(-0.99838585,3.06854867),
        std::complex<double>(-0.97476025,-0.90303271)
    };
    FourierTransformTester<double,1>::testSingle(ndarray::makeVector(5),xData1,kData1);
    double xData2[] = { -0.42299104, 0.12242535, 0.37497334, 0.47846245, 1.19236641, -1.54989674 };
    std::complex<double> kData2[] = {
        std::complex<double>(0.19533977,0.00000000),
        std::complex<double>(-2.39885906,-0.74039025),
        std::complex<double>(-0.01446277,-2.15615658),
        std::complex<double>(2.09335764,0.00000000),
    };
    FourierTransformTester<double,1>::testSingle(ndarray::makeVector(6),xData2,kData2);
    double xData3[] = {
        1.23233014,  0.80816934, -0.89026449,  0.64365113, -0.95490008,
        -2.31806551,  0.91599268,  0.48157162, -0.3389141 ,  1.3208367 ,
        0.24500594,  1.04908011, -0.47794142, -1.33640514, -0.49423689
    };
    std::complex<double> kData3[] = {
        std::complex<double>(0.83898605,0.00000000),
        std::complex<double>(1.38650224,-0.77516568),
        std::complex<double>(1.27483008,-2.49514665),
        std::complex<double>(0.06142139,0.00000000),
        std::complex<double>(-1.74225958,-0.09723986),
        std::complex<double>(-4.08361489,1.01828964),
        std::complex<double>(-1.01449740,0.00000000),
        std::complex<double>(1.88429912,-1.97237400),
        std::complex<double>(-0.76453557,-0.09069146),
    };
    FourierTransformTester<double,1>::testMultiplex(ndarray::makeVector(3,5),xData3,kData3);
    double xData4[] = {
        0.20943609, -0.70119705, -0.04658762, -0.31573777,  0.0884646 ,
        -0.53593713, -1.43900058, -0.60146942, -0.90063027, -1.85456039,
        1.82743304, -1.96060797, -1.13698948,  1.87844775, -0.20344457,
        -0.15419973,  0.11970772, -1.23579322
    };
    std::complex<double> kData4[] = {
        std::complex<double>(-1.30155887,0.00000000),
        std::complex<double>(-0.11433172,0.26007794),
        std::complex<double>(0.49132693,0.02616064),
        std::complex<double>(1.80418503,0.00000000),
        std::complex<double>(-4.92883560,0.00000000),
        std::complex<double>(-1.32888027,1.18552362),
        std::complex<double>(-2.47592366,-3.53962064),
        std::complex<double>(3.90443996,0.00000000),
        std::complex<double>(-0.73227153,0.00000000),
        std::complex<double>(-0.61959406,-2.41715370),
        std::complex<double>(-1.57064806,-2.97686989),
        std::complex<double>(-1.70918113,0.00000000),
    };
    FourierTransformTester<double,1>::testMultiplex(ndarray::makeVector(3,6),xData4,kData4);
}

BOOST_AUTO_TEST_CASE(complex_1d) {
    std::complex<double> xData1[] = {
        std::complex<double>(-1.43357821,0.68468601),
        std::complex<double>(0.03127727,0.04293266),
        std::complex<double>(-1.24543586,-1.09965670),
        std::complex<double>(0.62473820,1.02028337),
        std::complex<double>(1.24406733,-0.10372565),
    };
    std::complex<double> kData1[] = {
        std::complex<double>(-0.77893128,0.54451968),
        std::complex<double>(-1.64390926,2.98280693),
        std::complex<double>(-0.55477338,-0.35644026),
        std::complex<double>(-4.75954620,1.77512198),
        std::complex<double>(0.56926906,-1.52257831),
    };
    FourierTransformTester<std::complex<double>,1>::testSingle(ndarray::makeVector(5),xData1,kData1);
    std::complex<double> xData2[] = {
        std::complex<double>(1.11465199,0.04081224),
        std::complex<double>(-1.08053999,0.78927525),
        std::complex<double>(-2.05412193,-1.31663028),
        std::complex<double>(-0.12671457,0.16377679),
        std::complex<double>(2.01182088,-0.97858449),
        std::complex<double>(0.39217125,-0.92609902),
        std::complex<double>(-0.61199100,-0.16371873),
        std::complex<double>(-0.45056833,-1.23540146),
        std::complex<double>(0.18987288,0.07988060),
        std::complex<double>(0.43969961,-1.52021317),
        std::complex<double>(1.90211971,0.87354424),
        std::complex<double>(-2.12106807,-0.53018255),
        std::complex<double>(0.32101503,0.36269997),
        std::complex<double>(0.15932314,-0.08764936),
        std::complex<double>(-0.90755936,-1.64385208),
    };
    std::complex<double> kData2[] = {
        std::complex<double>(-0.13490361,-1.30135049),
        std::complex<double>(3.97794049,4.98890211),
        std::complex<double>(2.13438706,-0.17771391),
        std::complex<double>(-2.75975830,-0.14685547),
        std::complex<double>(2.35559432,-3.15892104),
        std::complex<double>(-0.04081559,-3.76555177),
        std::complex<double>(1.06683682,0.86503253),
        std::complex<double>(2.49923358,0.08822737),
        std::complex<double>(-1.59723642,0.07008247),
        std::complex<double>(0.03283785,-1.88828568),
        std::complex<double>(-0.64616954,-1.02543979),
        std::complex<double>(1.90149199,1.03828523),
        std::complex<double>(4.72705428,3.58443117),
        std::complex<double>(4.27447257,1.85030986),
        std::complex<double>(-0.74625074,-1.07986528),
    };
    FourierTransformTester<std::complex<double>,1>::testMultiplex(ndarray::makeVector(3,5),xData2,kData2);
}

BOOST_AUTO_TEST_CASE(real_2d) {
    double xData1[] = {
        0.24691759,  1.23546604,  0.11002248,  0.57353155, -0.25387112,
        -0.01661071,  0.77192489,  0.90072587, -0.39021137, -1.32054932,
        -1.57642169,  1.14263038
    };
    std::complex<double> kData1[] = {
        std::complex<double>(1.42355458,0.00000000),
        std::complex<double>(0.29730942,2.71858178),
        std::complex<double>(-3.60683302,0.00000000),
        std::complex<double>(2.53712918,-3.07155043),
        std::complex<double>(-1.28205146,-0.43653896),
        std::complex<double>(-0.37466977,-1.23205340),
        std::complex<double>(2.53712918,3.07155043),
        std::complex<double>(1.39542736,-4.26784629),
        std::complex<double>(-0.37466977,1.23205340),
    };
    FourierTransformTester<double,2>::testSingle(ndarray::makeVector(3,4),xData1,kData1);
    double xData2[] = {
        0.61726814,  1.02478783,  1.05838231, -0.54398019, -0.22777548,
        -0.52546761, -0.82971628,  1.95616799,  1.9787389 , -0.5170171 ,
        -1.0711432 , -1.35091919,  0.93562064, -0.90247789,  1.39481772
    };
    std::complex<double> kData2[] = {
        std::complex<double>(2.99728660,0.00000000),
        std::complex<double>(-4.76203006,-0.29149833),
        std::complex<double>(0.81503011,4.31207205),
        std::complex<double>(1.39438062,-2.64727323),
        std::complex<double>(1.99531720,-0.42062879),
        std::complex<double>(-2.98139538,-3.46555052),
        std::complex<double>(1.39438062,2.64727323),
        std::complex<double>(4.10890818,-5.68718353),
        std::complex<double>(2.56065709,1.51657561),
    };
    FourierTransformTester<double,2>::testSingle(ndarray::makeVector(3,5),xData2,kData2);
    double xData3[] = {
        -1.40706583, -0.10390223,  1.06058569, -0.31086132,  1.05681052,
        0.51060026,  0.94256284, -2.12855291, -0.44127652, -1.68077709,
        0.47362511, -0.06821889, -0.22098813, -0.21520295, -0.00649069,
        -1.15530166, -0.33289955,  0.49732109,  0.29713591,  0.95975404,
        -0.19866032,  3.40890222, -0.92217484, -0.2164837 ,  1.05600813,
        -1.0990793 ,  0.14508848,  0.20166354, -0.50525583,  0.40797194,
        1.16566663, -0.33373821, -0.39235002,  0.92020752, -0.35070829,
        0.91553651
    };
    std::complex<double> kData3[] = {
        std::complex<double>(-2.09647038,0.00000000),
        std::complex<double>(-3.26830548,-1.23355405),
        std::complex<double>(5.46695397,0.00000000),
        std::complex<double>(-0.09363035,-1.81698026),
        std::complex<double>(-5.74941461,-0.58493104),
        std::complex<double>(-2.63105188,-1.59000658),
        std::complex<double>(-0.09363035,1.81698026),
        std::complex<double>(1.61476551,1.19760785),
        std::complex<double>(-2.63105188,1.59000658),
        std::complex<double>(1.89491143,0.00000000),
        std::complex<double>(-0.12101837,-4.10305168),
        std::complex<double>(-4.66306664,0.00000000),
        std::complex<double>(-3.34443085,0.56315196),
        std::complex<double>(3.27891803,1.81358643),
        std::complex<double>(4.04607201,-2.44255096),
        std::complex<double>(-3.34443085,-0.56315196),
        std::complex<double>(-3.80139196,-0.53083089),
        std::complex<double>(4.04607201,2.44255096),
        std::complex<double>(2.13101109,0.00000000),
        std::complex<double>(-0.80164455,0.55436168),
        std::complex<double>(0.10588709,0.00000000),
        std::complex<double>(-0.60998428,0.31007277),
        std::complex<double>(1.12890713,3.08493192),
        std::complex<double>(3.09482501,-2.74095257),
        std::complex<double>(-0.60998428,-0.31007277),
        std::complex<double>(2.40549637,0.26293493),
        std::complex<double>(3.09482501,2.74095257),
    };
    FourierTransformTester<double,2>::testMultiplex(ndarray::makeVector(3,3,4),xData3,kData3);
    double xData4[] = {
        -2.02786312, -0.3539932 ,  0.55001132,  0.88336882,  0.43662574,
        2.01541172, -0.84654528,  0.01492169,  0.25803824, -0.94068415,
        -0.14705738, -1.14940977, -0.39805096,  0.83987327, -1.5626161 ,
        -1.42810026,  0.40392084,  0.49272843, -0.63469147, -0.44005778,
        -1.15325931, -0.27243829, -0.14669651,  0.43615407, -1.18495794,
        -0.48243325, -1.07236569,  0.96292895, -2.46877351, -1.34336383,
        -0.66991209, -1.19233445, -0.62648047, -0.52768348,  0.23126765,
        1.76633904, -0.65569182,  0.29814291, -0.09862708,  0.98570324,
        0.0580575 , -0.46466068,  0.35672026,  0.31357366,  0.22521739
    };
    std::complex<double> kData4[] = {
        std::complex<double>(-2.42796916,0.00000000),
        std::complex<double>(-3.26222014,1.33588590),
        std::complex<double>(4.07743277,-1.55909118),
        std::complex<double>(0.44620892,-2.52741129),
        std::complex<double>(-3.35542310,-1.48469802),
        std::complex<double>(-3.53459988,-0.17856927),
        std::complex<double>(0.44620892,2.52741129),
        std::complex<double>(-2.86822821,2.99240988),
        std::complex<double>(-5.49815917,2.18067744),
        std::complex<double>(-8.33140554,0.00000000),
        std::complex<double>(-3.17289317,-4.26545758),
        std::complex<double>(-0.32088611,2.58992671),
        std::complex<double>(1.75640242,-1.80376581),
        std::complex<double>(1.11499523,1.51702830),
        std::complex<double>(-5.63675876,0.34121561),
        std::complex<double>(1.75640242,1.80376581),
        std::complex<double>(-1.91535212,-1.64762718),
        std::complex<double>(1.62944330,-1.20265674),
        std::complex<double>(-0.00036843,0.00000000),
        std::complex<double>(1.11553297,3.37059349),
        std::complex<double>(1.77086237,2.53148430),
        std::complex<double>(-4.17753005,-1.56487168),
        std::complex<double>(-0.00380698,-1.52895218),
        std::complex<double>(-0.48338932,-1.10597651),
        std::complex<double>(-4.17753005,1.56487168),
        std::complex<double>(-1.21120542,2.39435110),
        std::complex<double>(-2.03462006,0.80292460),
    };
    FourierTransformTester<double,2>::testMultiplex(ndarray::makeVector(3,3,5),xData4,kData4);
}

BOOST_AUTO_TEST_CASE(complex_2d) {
    std::complex<double> xData1[] = {
        std::complex<double>(-0.92330789,-2.17834411),
        std::complex<double>(0.66032244,-0.73829353),
        std::complex<double>(-0.68029780,0.13129172),
        std::complex<double>(0.23413066,0.07680841),
        std::complex<double>(-0.44237642,0.55046543),
        std::complex<double>(1.74734692,0.15625388),
        std::complex<double>(1.00639051,-0.27572242),
        std::complex<double>(-1.84657822,-1.11254821),
        std::complex<double>(-0.68120146,-1.25764975),
        std::complex<double>(1.27376530,0.17496170),
        std::complex<double>(0.50483156,0.90637023),
        std::complex<double>(0.17423552,1.38104926),
    };
    std::complex<double> kData1[] = {
        std::complex<double>(1.02726111,-2.18535738),
        std::complex<double>(-3.63019744,-8.76711466),
        std::complex<double>(-3.45918412,-2.06182040),
        std::complex<double>(-2.12542261,1.47217875),
        std::complex<double>(-3.21092823,-2.27137660),
        std::complex<double>(0.65731688,-1.63596716),
        std::complex<double>(0.70037550,-3.02857437),
        std::complex<double>(6.67065478,-1.19040390),
        std::complex<double>(0.05620935,-3.66887857),
        std::complex<double>(-0.20145550,2.19559898),
        std::complex<double>(-4.73536776,0.93369296),
        std::complex<double>(-2.82895659,-5.93210699),
    };
    FourierTransformTester<std::complex<double>,2>::testSingle(ndarray::makeVector(3,4),xData1,kData1);
    std::complex<double> xData2[] = {
        std::complex<double>(0.19529047,1.20435172),
        std::complex<double>(1.45906066,-0.97965946),
        std::complex<double>(-1.83174935,-0.36226429),
        std::complex<double>(0.11703370,-0.87754715),
        std::complex<double>(-0.52421348,0.93475249),
        std::complex<double>(-1.20346126,0.29011118),
        std::complex<double>(-1.48942112,-1.54360504),
        std::complex<double>(-0.42518103,-1.43737264),
        std::complex<double>(1.06970149,-0.99337823),
        std::complex<double>(-0.03119172,-0.05324573),
        std::complex<double>(-0.26718251,0.02763509),
        std::complex<double>(0.04403172,0.33843024),
        std::complex<double>(-0.27862895,0.11800338),
        std::complex<double>(-0.45012170,0.59134803),
        std::complex<double>(1.79455284,1.00858644),
        std::complex<double>(0.90559524,-0.70387881),
        std::complex<double>(-1.13560698,-0.07159557),
        std::complex<double>(0.17003962,1.91449110),
        std::complex<double>(-1.09018239,-0.14139606),
        std::complex<double>(-0.63102161,0.95133602),
        std::complex<double>(3.40992650,-0.77678722),
        std::complex<double>(-0.70506997,2.22309013),
        std::complex<double>(0.79939137,-0.43430236),
        std::complex<double>(1.25702706,-0.40232699),
        std::complex<double>(-0.05621207,-0.82483426),
        std::complex<double>(-0.45139484,-2.49048884),
        std::complex<double>(-0.58693292,0.05053400),
        std::complex<double>(-0.53649309,1.79843410),
        std::complex<double>(-0.29692612,0.16872552),
        std::complex<double>(1.07050210,-0.58501773),
        std::complex<double>(-0.17909989,-1.51092040),
        std::complex<double>(-0.77482019,-0.71248210),
        std::complex<double>(0.33169239,-0.33502900),
        std::complex<double>(-1.48887694,-2.10226979),
        std::complex<double>(0.17078066,0.72966237),
        std::complex<double>(0.83173292,-1.19673034),
    };
    std::complex<double> kData2[] = {
        std::complex<double>(-2.88728243,-3.45179183),
        std::complex<double>(5.56282699,2.53543695),
        std::complex<double>(-2.80786657,1.98677531),
        std::complex<double>(3.09543592,3.51248350),
        std::complex<double>(0.42163615,4.06364305),
        std::complex<double>(3.74538684,-2.44419993),
        std::complex<double>(-1.86528567,4.07284857),
        std::complex<double>(4.06768924,4.76385009),
        std::complex<double>(2.28455274,-3.65720875),
        std::complex<double>(-3.53343134,0.58253015),
        std::complex<double>(-4.96450748,2.03825824),
        std::complex<double>(-0.77566878,0.44959533),
        std::complex<double>(4.04590102,4.27656807),
        std::complex<double>(5.37572779,1.35348531),
        std::complex<double>(2.95300374,-4.87155087),
        std::complex<double>(-4.39187028,-3.68002016),
        std::complex<double>(2.70357575,5.83300183),
        std::complex<double>(-5.89071199,3.76064790),
        std::complex<double>(0.07349988,8.99019172),
        std::complex<double>(-0.10666298,-0.66887248),
        std::complex<double>(-0.83528451,-7.06739279),
        std::complex<double>(-1.81888066,-3.71873158),
        std::complex<double>(0.15484744,-0.40127904),
        std::complex<double>(-5.60669264,-2.39000736),
        std::complex<double>(-1.96604800,-7.01041645),
        std::complex<double>(-4.49319166,0.12977561),
        std::complex<double>(0.73265208,3.56669295),
        std::complex<double>(5.64080434,-0.65060303),
        std::complex<double>(-1.23431266,1.32790932),
        std::complex<double>(-4.62184457,-2.15880098),
        std::complex<double>(-3.08669912,-0.23413824),
        std::complex<double>(10.39353240,0.27589821),
        std::complex<double>(-1.69273809,1.28344216),
        std::complex<double>(-2.15957004,-0.85237418),
        std::complex<double>(3.38827586,-3.57929124),
        std::complex<double>(-1.57540539,-1.99610520),
    };
    FourierTransformTester<std::complex<double>,2>::testMultiplex(ndarray::makeVector(3,3,4),xData2,kData2);
};

template <typename T, int N>
struct FourierOpsTester {
    typedef ndarray::FourierTransform<T,N> FFT;

    static void makeGaussian(T const * mu, T sigma, typename FFT::ArrayX const & array, int m=0) {
        typename FFT::ArrayX::Iterator const iter_end = array.end();
        typename FFT::ArrayX::Iterator iter = array.begin();
        for (int x=0; iter != iter_end; ++iter, ++x) {
            T z = (x-(*mu))/sigma;
            (*iter) *= std::exp(-0.5*z*z);
            if (m == N) (*iter) *= -z / sigma;
            FourierOpsTester<T,N-1>::makeGaussian(mu+1,sigma,*iter,m);
        }
    }

    static void testShift(
        typename FFT::Index const & shape,
        ndarray::Vector<T,N> const & offset,
        T sigma
    ) {
        typename FFT::ArrayX x1;
        typename FFT::ArrayX x2;
        typename FFT::ArrayK k;
        typename FFT::Ptr forward = FFT::planForward(shape,x1,k);
        typename FFT::Ptr inverse = FFT::planInverse(shape,k,x2);
        x1.deep() = 1.0;
        ndarray::Vector<T,N> mu = (shape - offset) * 0.5;
        makeGaussian(mu.begin(), sigma, x1);
        forward->execute();
        ndarray::shift(offset, k, shape[N-1]);
        inverse->execute();
        x2.deep() /= shape.product();
        x1.deep() = 1.0;
        mu += offset;
        makeGaussian(mu.begin(), sigma, x1);
        BOOST_CHECK(compareAbsolute(x1, x2));
    }

    static void testDifferentiate(
        typename FFT::Index const & shape,
        T sigma,
        int n
    ) {
        typename FFT::ArrayX x1;
        typename FFT::ArrayX x2;
        typename FFT::ArrayK k;
        typename FFT::Ptr forward = FFT::planForward(shape,x1,k);
        typename FFT::Ptr inverse = FFT::planInverse(shape,k,x2);
        x1.deep() = 1.0;
        ndarray::Vector<T,N> mu = shape * 0.5;
        makeGaussian(mu.begin(),sigma,x1,0);
        forward->execute();
        ndarray::differentiate(n, k, shape[N-1]);
        inverse->execute();
        x2.deep() /= shape.product();
        x1.deep() = 1.0;
        makeGaussian(mu.begin(), sigma, x1, N-n);
        BOOST_CHECK(compareAbsolute(x1,x2));
    }

};

template <typename T>
struct FourierOpsTester<T,0> {

    static void makeGaussian(T const * mu, T sigma, T value, int m=0) {}

};

BOOST_AUTO_TEST_CASE(ops) {
    FourierOpsTester<double,1>::testShift(ndarray::makeVector(256),ndarray::makeVector(7.25),10.0);
    FourierOpsTester<double,1>::testShift(ndarray::makeVector(255),ndarray::makeVector(7.25),10.0);
    FourierOpsTester<double,2>::testShift(ndarray::makeVector(256,256),ndarray::makeVector(7.25,6.4),10.0);
    FourierOpsTester<double,2>::testShift(ndarray::makeVector(256,255),ndarray::makeVector(7.25,6.4),10.0);

    FourierOpsTester<double,1>::testDifferentiate(ndarray::makeVector(256),10.0,0);
    FourierOpsTester<double,1>::testDifferentiate(ndarray::makeVector(256),10.0,1);
    FourierOpsTester<double,1>::testDifferentiate(ndarray::makeVector(255),10.0,0);
    FourierOpsTester<double,1>::testDifferentiate(ndarray::makeVector(255),10.0,1);
    FourierOpsTester<double,2>::testDifferentiate(ndarray::makeVector(256,256),10.0,0);
    FourierOpsTester<double,2>::testDifferentiate(ndarray::makeVector(256,256),10.0,1);
    FourierOpsTester<double,2>::testDifferentiate(ndarray::makeVector(256,255),10.0,0);
    FourierOpsTester<double,2>::testDifferentiate(ndarray::makeVector(256,255),10.0,1);
}

#else

BOOST_AUTO_TEST_CASE(placeholder) {

    std::cerr << "WARNING: ndarray-fft test code disabled on gcc 4.5.\n";

}

#endif
