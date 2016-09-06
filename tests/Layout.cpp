#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <array>

#include "ndarray/detail/Layout.hpp"

namespace ndd = ndarray::detail;

TEST_CASE( "Layout (in)equality comparison", "[layout-equality]" ) {
    std::array<int,4> shape = { 3, 5, 4, 2 };
    std::array<int,4> strides = { 160, 32, 8, 4 };
    auto a = ndd::Layout<4>::make(shape, strides);
    CHECK( *a == *ndd::Layout<4>::make(shape, strides) );
    CHECK( *a != *ndd::Layout<4>::make(shape, std::array<int,4>{160, 32, 8, 2}) );
    CHECK( *a != *ndd::Layout<4>::make(shape, std::array<int,4>{320, 32, 8, 4}) );
    CHECK( *a != *ndd::Layout<4>::make(std::array<int,4>{3, 5, 4, 1} , strides) );
    CHECK( *a != *ndd::Layout<4>::make(std::array<int,4>{6, 5, 4, 2} , strides) );
}

TEST_CASE( "Layout accessors and casting", "[layout-accessors]" ) {
    std::array<int,4> shape = { 3, 5, 4, 2 };
    std::array<int,4> strides = { 160, 32, 8, 4 };
    auto a = ndd::Layout<4>::make(shape, strides);
    CHECK( a->size() == 3 );
    CHECK( a->stride() == 160 );
    CHECK( a->full_size() == 2*4*5*3 );
    CHECK( ndd::get_dim<0>(*a).size() == 3 );
    CHECK( ndd::get_dim<0>(*a).stride() == 160 );
    CHECK( ndd::get_dim<0>(*a).full_size() == 2*4*5*3 );
    CHECK( ndd::get_dim<1>(*a).size() == 5 );
    CHECK( ndd::get_dim<1>(*a).stride() == 32 );
    CHECK( ndd::get_dim<1>(*a).full_size() == 2*4*5 );
    CHECK( ndd::get_dim<2>(*a).size() == 4 );
    CHECK( ndd::get_dim<2>(*a).stride() == 8 );
    CHECK( ndd::get_dim<2>(*a).full_size() == 2*4 );
    CHECK( ndd::get_dim<3>(*a).size() == 2 );
    CHECK( ndd::get_dim<3>(*a).stride() == 4 );
    CHECK( ndd::get_dim<3>(*a).full_size() == 2 );
}

TEST_CASE( "Layout with row-major strides", "[layout-row-major]" ) {
    std::array<int,4> shape = { 3, 5, 4, 2 };
    std::array<int,4> strides = { 40*sizeof(int), 8*sizeof(int), 2*sizeof(int), sizeof(int) };
    auto a = ndd::Layout<4>::make(shape, strides);
    auto b = ndd::Layout<4>::make(shape, 4, ndarray::MemoryOrder::ROW_MAJOR);
    CHECK( *a == *b );
    CHECK_NOTHROW( (ndd::check_contiguousness<4,4>(*b, sizeof(int))) );
    CHECK_NOTHROW( (ndd::check_contiguousness<4,3>(*b, sizeof(int))) );
    CHECK_NOTHROW( (ndd::check_contiguousness<4,2>(*b, sizeof(int))) );
    CHECK_NOTHROW( (ndd::check_contiguousness<4,1>(*b, sizeof(int))) );
    CHECK_NOTHROW( (ndd::check_contiguousness<4,0>(*b, sizeof(int))) );
    CHECK_THROWS_AS( (ndd::check_contiguousness<4,-1>(*b, sizeof(int))),
                     ndarray::NoncontiguousError );
    CHECK_THROWS_AS( (ndd::check_contiguousness<4,-2>(*b, sizeof(int))),
                     ndarray::NoncontiguousError );
    CHECK_THROWS_AS( (ndd::check_contiguousness<4,-3>(*b, sizeof(int))),
                     ndarray::NoncontiguousError );
    CHECK_THROWS_AS( (ndd::check_contiguousness<4,-4>(*b, sizeof(int))),
                     ndarray::NoncontiguousError );
}

TEST_CASE( "Layout with col-major strides", "[layout-col-major]" ) {
    std::array<int,4> shape = { 3, 5, 4, 2 };
    std::array<int,4> strides = { sizeof(int), 3*sizeof(int), 15*sizeof(int), 60*sizeof(int) };
    auto a = ndd::Layout<4>::make(shape, strides);
    auto b = ndd::Layout<4>::make(shape, sizeof(int), ndarray::MemoryOrder::COL_MAJOR);
    CHECK( *a == *b );
    CHECK_NOTHROW( (ndd::check_contiguousness<4,-4>(*b, sizeof(int))) );
    CHECK_NOTHROW( (ndd::check_contiguousness<4,-3>(*b, sizeof(int))) );
    CHECK_NOTHROW( (ndd::check_contiguousness<4,-2>(*b, sizeof(int))) );
    CHECK_NOTHROW( (ndd::check_contiguousness<4,-1>(*b, sizeof(int))) );
    CHECK_NOTHROW( (ndd::check_contiguousness<4,0>(*b, sizeof(int))) );
    CHECK_THROWS_AS( (ndd::check_contiguousness<4,1>(*b, sizeof(int))),
                     ndarray::NoncontiguousError );
    CHECK_THROWS_AS( (ndd::check_contiguousness<4,2>(*b, sizeof(int))),
                     ndarray::NoncontiguousError );
    CHECK_THROWS_AS( (ndd::check_contiguousness<4,3>(*b, sizeof(int))),
                     ndarray::NoncontiguousError );
    CHECK_THROWS_AS( (ndd::check_contiguousness<4,4>(*b, sizeof(int))),
                     ndarray::NoncontiguousError );
}
