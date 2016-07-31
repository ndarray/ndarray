#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <regex>

#include "ndarray/formatting/types.hpp"
#include "ndarray/formatting/Vector.hpp"

namespace ndd = ndarray::detail;
namespace nd = ndarray;

TEST_CASE(
    "Test stringification of types",
    "[formatting-types]"
) {
    CHECK(
        std::regex_match(
            nd::type_string<int const &>(),
            std::regex(
                "(int const)|(const int) ?&",
                std::regex_constants::egrep
            )
        )
    );
    CHECK(
        std::regex_match(
            nd::type_string<ndarray::Array<float,2>>(),
            std::regex(
                "ndarray::Array<float, ?2u?l?>",
                std::regex_constants::egrep
            )
        )
    );
}


TEST_CASE(
    "Test stringification of Vectors",
    "[formatting-vector]"
) {
    std::ostringstream oss;
    oss << nd::Vector<int,3>{4, 5, 6};
    CHECK(oss.str() == "Vector<int, 3>{4, 5, 6}");
}
