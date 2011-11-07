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
#include "lsst/ndarray/tables.h"

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE ndarray-tables
#include "boost/test/included/unit_test.hpp"
#include "boost/fusion/container/vector.hpp"
#include "boost/make_shared.hpp"
#include "boost/random/mersenne_twister.hpp"
#include "boost/random/variate_generator.hpp"
#include "boost/random/normal_distribution.hpp"

typedef boost::mt19937 RandomEngine;
typedef boost::variate_generator< RandomEngine &, boost::normal_distribution<double> > RandomGenerator;

static RandomEngine random_engine;
static RandomGenerator random_generator(random_engine, boost::normal_distribution<double>(0.0, 5.0));

namespace nt = lsst::ndarray::tables;

struct Tag {
    typedef boost::fusion::vector< nt::Field<int>, nt::Field<double,2>, nt::Field<float,1> > FieldSequence;

    static const nt::Index<0> a;
    static const nt::Index<1> b;
    static const nt::Index<2> c;
};

static void fillLayout(nt::Layout<Tag> & layout) {
    layout[Tag::a].name = "a";
    layout[Tag::b].name = "b";
    layout[Tag::b].shape = lsst::ndarray::makeVector(3, 2);
    layout[Tag::c].name = "c";
    layout[Tag::c].shape = lsst::ndarray::makeVector(5);
}

static void randomize(nt::Row<Tag> const & row) {
    row[Tag::a] = random_generator();
    for (int i = 0; i < row[Tag::b].getSize<0>(); ++i) {
        for (int j = 0; j < row[Tag::b].getSize<1>(); ++j) {
            row[Tag::b][i][j] = random_generator();
        }
    }
    for (int i = 0; i < row[Tag::c].getSize<0>(); ++i) {
        row[Tag::c][i] = random_generator();
    }
}

BOOST_AUTO_TEST_CASE(layout) {

    nt::Layout<Tag> layout1;
    fillLayout(layout1);
    layout1.normalize(true);

    nt::Layout<Tag> layout2;
    fillLayout(layout2);
    layout2.normalize(false);

    std::cerr << "bytes: " << layout1.getBytes() << ", offsets: " 
              << layout1[Tag::a].offset << ", "
              << layout1[Tag::b].offset << ", "
              << layout1[Tag::c].offset << "\n";

    std::cerr << "bytes: " << layout2.getBytes() << ", offsets: " 
              << layout2[Tag::a].offset << ", "
              << layout2[Tag::b].offset << ", "
              << layout2[Tag::c].offset << "\n";

}

BOOST_AUTO_TEST_CASE(row) {

    nt::Layout<Tag> layout1;
    fillLayout(layout1);

    nt::Row<Tag> row1 = nt::Row<Tag>::allocate(layout1);
    randomize(row1);

    std::cerr << "a: " << row1[Tag::a] << "\n";
    std::cerr << "b: " << row1[Tag::b] << "\n";
    std::cerr << "c: " << row1[Tag::c] << "\n";

    BOOST_CHECK( row1.getLayout() == layout1 );

    {
        nt::Row<Tag> row2 = row1;
        BOOST_CHECK( row2.getLayout() == row1.getLayout() );
        BOOST_CHECK( row2[Tag::a] == row1[Tag::a] );
        BOOST_CHECK( &row2[Tag::a] == &row1[Tag::a] );
        BOOST_CHECK( row2[Tag::b].shallow() == row1[Tag::b].shallow() );
        BOOST_CHECK( row2[Tag::c].shallow() == row1[Tag::c].shallow() );
    }

    {
        nt::Row<Tag const> row2 = row1;
        BOOST_CHECK( row2.getLayout() == row1.getLayout() );
        BOOST_CHECK( row2[Tag::a] == row1[Tag::a] );
        BOOST_CHECK( &row2[Tag::a] == &row1[Tag::a] );
        BOOST_CHECK( row2[Tag::b].shallow() == row1[Tag::b].shallow() );
        BOOST_CHECK( row2[Tag::c].shallow() == row1[Tag::c].shallow() );
    }

    {
        nt::Row<Tag> row2 = nt::Row<Tag>::allocate(layout1);
        row2 = row1;
        BOOST_CHECK( row2.getLayout() == row1.getLayout() );
        BOOST_CHECK( row2[Tag::a] == row1[Tag::a] );
        BOOST_CHECK( &row2[Tag::a] == &row1[Tag::a] );
        BOOST_CHECK( row2[Tag::b].shallow() == row1[Tag::b].shallow() );
        BOOST_CHECK( row2[Tag::c].shallow() == row1[Tag::c].shallow() );
    }

    {
        nt::Row<Tag const> row2 = nt::Row<Tag>::allocate(layout1);
        row2 = row1;
        BOOST_CHECK( row2.getLayout() == row1.getLayout() );
        BOOST_CHECK( row2[Tag::a] == row1[Tag::a] );
        BOOST_CHECK( &row2[Tag::a] == &row1[Tag::a] );
        BOOST_CHECK( row2[Tag::b].shallow() == row1[Tag::b].shallow() );
        BOOST_CHECK( row2[Tag::c].shallow() == row1[Tag::c].shallow() );
    }

#ifdef FAILTEST_01 // should fail to compile
    {
        nt::Row<Tag const> rowc(row1);
        nt::Row<Tag> row2(rowc);
    }
#endif

#ifdef FAILTEST_02 // should fail to compile
    {
        nt::Row<Tag const> rowc(row1);
        nt::Row<Tag> row2 = nt::Row<Tag const>::allocate(tag1.fields);
        row2 = rowc;
    }
#endif

}

BOOST_AUTO_TEST_CASE(tables) {

    nt::Layout<Tag> layout1;
    fillLayout(layout1);
    nt::Table<Tag> table1 = nt::Table<Tag>::allocate(5, layout1);

    BOOST_CHECK( &table1[0][Tag::a] == &table1[Tag::a][0] );
    BOOST_CHECK( &table1[1][Tag::a] == &table1[Tag::a][1] );
    BOOST_CHECK( table1[Tag::b][0].shallow() == table1[0][Tag::b].shallow() );
    BOOST_CHECK( table1[Tag::b][1].shallow() == table1[1][Tag::b].shallow() );
    BOOST_CHECK( table1[Tag::c][0].shallow() == table1[0][Tag::c].shallow() );
    BOOST_CHECK( table1[Tag::c][1].shallow() == table1[1][Tag::c].shallow() );

    nt::Table<Tag>::Iterator i = table1.begin();
    nt::Table<Tag>::Iterator const end = table1.end();
    for (int n = 0; i != end; ++i, ++n) {
        randomize(*i);
        BOOST_CHECK( &(*i)[Tag::a] == &table1[n][Tag::a] );
        BOOST_CHECK( (*i)[Tag::b].shallow() == table1[n][Tag::b].shallow() );
        BOOST_CHECK( (*i)[Tag::c].shallow() == table1[n][Tag::c].shallow() );
    }

    std::cerr << "a:\n" << table1[Tag::a] << "\n";
    std::cerr << "b:\n" << table1[Tag::b] << "\n";
    std::cerr << "c:\n" << table1[Tag::c] << "\n";

    BOOST_CHECK( table1.getLayout() == layout1 );

}
