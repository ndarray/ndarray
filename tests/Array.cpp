#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "ndarray/Array.hpp"

namespace ndd = ndarray::detail;
namespace nd = ndarray;

TEST_CASE(
    "default-constructed and post-move Arrays behave appropriately",
    "[array-empty]"
) {
    nd::Array<float,3> empty;
    CHECK( empty.data() == nullptr );
    CHECK( empty.manager() == nullptr );
    CHECK( empty.empty() );
    nd::Array<float,3> b;
    CHECK( empty == b );
    nd::Array<float,3> full({4, 3, 2});
    CHECK( empty != full );
    empty = std::move(full);
    // No check for data()==nullptr; this is not guaranteed.
    CHECK( full.empty() );
    CHECK( full.manager() == nullptr );
}

TEST_CASE(
    "Arrays can use different container types for shapes and strides",
    "[array-primary-ctors]"
) {
    // initializer_list for both.
    nd::Array<float,3> a(nullptr, {4, 5, 2}, {40, 8, 4});
    CHECK( a.data() == nullptr );
    CHECK( a.size<0>() == 4 );
    CHECK( a.size<1>() == 5 );
    CHECK( a.size<2>() == 2 );
    CHECK( a.stride<0>() == 40 );
    CHECK( a.stride<1>() == 8 );
    CHECK( a.stride<2>() == 4 );
    // initializer_list for shape, std::vector for strides
    nd::Array<float,3> b(nullptr, {4, 5, 2}, std::vector<int>{40, 8, 4});
    CHECK( b.data() == nullptr );
    CHECK( b.size<0>() == 4 );
    CHECK( b.size<1>() == 5 );
    CHECK( b.size<2>() == 2 );
    CHECK( b.stride<0>() == 40 );
    CHECK( b.stride<1>() == 8 );
    CHECK( b.stride<2>() == 4 );
    // std::array for shape, initializer_list for strides
    nd::Array<float,3> c(nullptr, std::array<unsigned char,3>{4, 5, 2},
                         {40, 8, 4});
    CHECK( c.data() == nullptr );
    CHECK( c.size<0>() == 4 );
    CHECK( c.size<1>() == 5 );
    CHECK( c.size<2>() == 2 );
    CHECK( c.stride<0>() == 40 );
    CHECK( c.stride<1>() == 8 );
    CHECK( c.stride<2>() == 4 );
    // std::vector for shape, std::array for strides
    nd::Array<float,3> d(nullptr, std::vector<unsigned char>{4, 5, 2},
                         std::array<int,3>{40, 8, 4});
    CHECK( d.data() == nullptr );
    CHECK( d.size<0>() == 4 );
    CHECK( d.size<1>() == 5 );
    CHECK( d.size<2>() == 2 );
    CHECK( d.stride<0>() == 40 );
    CHECK( d.stride<1>() == 8 );
    CHECK( d.stride<2>() == 4 );
    // std::vector for shape, automatic row-major strides
    nd::Array<float,3> e(std::vector<short>{4, 5, 2});
    CHECK( e.data() != nullptr );
    CHECK( e.size<0>() == 4 );
    CHECK( e.size<1>() == 5 );
    CHECK( e.size<2>() == 2 );
    CHECK( e.stride<0>() == 40 );
    CHECK( e.stride<1>() == 8 );
    CHECK( e.stride<2>() == 4 );
    // initializer_list for shape, automatic row-major strides
    nd::Array<float,3> f({4, 5, 2});
    CHECK( f.data() != nullptr );
    CHECK( f.size<0>() == 4 );
    CHECK( f.size<1>() == 5 );
    CHECK( f.size<2>() == 2 );
    CHECK( f.stride<0>() == 40 );
    CHECK( f.stride<1>() == 8 );
    CHECK( f.stride<2>() == 4 );
    // std::array for shape, automatic column-major strides
    nd::Array<float,3> g(std::array<long,3>{4, 5, 2},
                         nd::MemoryOrder::COL_MAJOR);
    CHECK( g.data() != nullptr );
    CHECK( g.size<0>() == 4 );
    CHECK( g.size<1>() == 5 );
    CHECK( g.size<2>() == 2 );
    CHECK( g.stride<0>() == 4 );
    CHECK( g.stride<1>() == 16 );
    CHECK( g.stride<2>() == 80 );
    // initializer_list for shape, automatic column-major strides
    nd::Array<float,3> h({4, 5, 2}, nd::MemoryOrder::COL_MAJOR);
    CHECK( h.data() != nullptr );
    CHECK( h.size<0>() == 4 );
    CHECK( h.size<1>() == 5 );
    CHECK( h.size<2>() == 2 );
    CHECK( h.stride<0>() == 4 );
    CHECK( h.stride<1>() == 16 );
    CHECK( h.stride<2>() == 80 );
}

TEST_CASE(
    "Arrays equality comparison is shallow and requires exact equivalence.",
    "[array-comparison]"
) {
    nd::Array<float,3> a({4, 5, 2});
    nd::Array<float,3> b({4, 5, 2});
    CHECK( a != b );  // different data, same shape and strides
    nd::Array<float,3> c(a);
    CHECK( a == c );
    b = a;
    CHECK( a == b );
    nd::Array<float,3> d(a.data(), a.shape(), nd::MemoryOrder::COL_MAJOR,
                         a.manager(), a.dtype());
    CHECK( a != d ); // same data, shape; different strides
    nd::Array<float,3> e(a.data(), a.shape().reversed(), a.strides(),
                         a.manager(), a.dtype());
    CHECK( a != e ); // same data, strides; different shape
}
}
