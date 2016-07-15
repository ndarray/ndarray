#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "ndarray/Array.hpp"

namespace ndd = ndarray::detail;
namespace nd = ndarray;

TEST_CASE( "default-constructed Arrays behave appropriately", "[array-default-ctor]" ) {
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
