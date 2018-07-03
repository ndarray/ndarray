cmake_minimum_required(VERSION 3.5)
project(ndarray VERSION 2.0.0 LANGUAGES CXX)

find_package(fmt 4.1 REQUIRED)
find_package(Catch2 2.2.3 REQUIRED)

add_library(ndarray INTERFACE)
target_include_directories(ndarray
    INTERFACE
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_compile_features(ndarray
    INTERFACE
        cxx_auto_type
        cxx_constexpr
        cxx_deleted_functions
        cxx_generic_lambdas
        cxx_noexcept
        cxx_static_assert
        cxx_strong_enums
)
target_link_libraries(ndarray INTERFACE fmt::fmt)

enable_testing()

add_executable(tests
    tests/main.cpp
    tests/Layout.cpp
    tests/ArrayImpl.cpp
)
target_link_libraries(tests Catch2::Catch2 ndarray)
include(Catch)
catch_discover_tests(tests)
