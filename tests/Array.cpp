#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "ndarray/Array.hpp"

namespace ndd = ndarray::detail;
namespace nd = ndarray;

TEST_CASE(
    "default-constructed Arrays behave appropriately",
    "[array-default-ctor]"
) {
    nd::Array<float,3> empty;
    REQUIRE( empty.data() == nullptr );
    REQUIRE( empty.manager() == nullptr );
    REQUIRE( empty.stride() == 0 );
    REQUIRE( empty.size() == 0);
    REQUIRE( empty.full_size() == 0);
    REQUIRE( empty.stride<0>() == 0 );
    REQUIRE( empty.size<0>() == 0);
    REQUIRE( empty.stride<1>() == 0 );
    REQUIRE( empty.size<1>() == 0);
    REQUIRE( empty.stride<2>() == 0 );
    REQUIRE( empty.size<2>() == 0);
    REQUIRE( std::distance(empty.begin(), empty.end()) == 0 );
}

TEST_CASE(
    "Arrays can use different container types for shapes and strides",
    "[array-primary-ctors]"
) {
    // initializer_list for both.
    nd::Array<float,3> a({4, 5, 2}, {40, 8, 4});
    REQUIRE( a.size<0>() == 4 );
    REQUIRE( a.size<1>() == 5 );
    REQUIRE( a.size<2>() == 2 );
    REQUIRE( a.stride<0>() == 40 );
    REQUIRE( a.stride<1>() == 8 );
    REQUIRE( a.stride<2>() == 4 );
    // initializer_list for shape, std::vector for strides
    nd::Array<float,3> b({4, 5, 2}, std::vector<int>{40, 8, 4});
    REQUIRE( b.size<0>() == 4 );
    REQUIRE( b.size<1>() == 5 );
    REQUIRE( b.size<2>() == 2 );
    REQUIRE( b.stride<0>() == 40 );
    REQUIRE( b.stride<1>() == 8 );
    REQUIRE( b.stride<2>() == 4 );
    // std::array for shape, initializer_list for strides
    nd::Array<float,3> c(std::array<unsigned char,3>{4, 5, 2}, {40, 8, 4});
    REQUIRE( c.size<0>() == 4 );
    REQUIRE( c.size<1>() == 5 );
    REQUIRE( c.size<2>() == 2 );
    REQUIRE( c.stride<0>() == 40 );
    REQUIRE( c.stride<1>() == 8 );
    REQUIRE( c.stride<2>() == 4 );
    // std::vector for shape, std::array for strides
    nd::Array<float,3> d(std::vector<unsigned char>{4, 5, 2},
                         std::array<int,3>{40, 8, 4});
    REQUIRE( d.size<0>() == 4 );
    REQUIRE( d.size<1>() == 5 );
    REQUIRE( d.size<2>() == 2 );
    REQUIRE( d.stride<0>() == 40 );
    REQUIRE( d.stride<1>() == 8 );
    REQUIRE( d.stride<2>() == 4 );
    // std::vector for shape, automatic row-major strides
    nd::Array<float,3> e(std::vector<short>{4, 5, 2});
    REQUIRE( e.size<0>() == 4 );
    REQUIRE( e.size<1>() == 5 );
    REQUIRE( e.size<2>() == 2 );
    REQUIRE( e.stride<0>() == 40 );
    REQUIRE( e.stride<1>() == 8 );
    REQUIRE( e.stride<2>() == 4 );
    // initializer_list for shape, automatic row-major strides
    nd::Array<float,3> f({4, 5, 2});
    REQUIRE( f.size<0>() == 4 );
    REQUIRE( f.size<1>() == 5 );
    REQUIRE( f.size<2>() == 2 );
    REQUIRE( f.stride<0>() == 40 );
    REQUIRE( f.stride<1>() == 8 );
    REQUIRE( f.stride<2>() == 4 );
    // std::array for shape, automatic column-major strides
    nd::Array<float,3> g(std::array<long,3>{4, 5, 2},
                         nd::MemoryOrder::COL_MAJOR);
    REQUIRE( g.size<0>() == 4 );
    REQUIRE( g.size<1>() == 5 );
    REQUIRE( g.size<2>() == 2 );
    REQUIRE( g.stride<0>() == 4 );
    REQUIRE( g.stride<1>() == 16 );
    REQUIRE( g.stride<2>() == 80 );
    // initializer_list for shape, automatic column-major strides
    nd::Array<float,3> h({4, 5, 2}, nd::MemoryOrder::COL_MAJOR);
    REQUIRE( h.size<0>() == 4 );
    REQUIRE( h.size<1>() == 5 );
    REQUIRE( h.size<2>() == 2 );
    REQUIRE( h.stride<0>() == 4 );
    REQUIRE( h.stride<1>() == 16 );
    REQUIRE( h.stride<2>() == 80 );
}
