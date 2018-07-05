cmake_minimum_required(VERSION 3.5)
project(ndarray VERSION 2.0.0 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)  # used to obtain build commands for expected compile failure tests

find_package(fmt 4.1 REQUIRED)
find_package(Catch2 2.2.3 REQUIRED)
find_package(pybind11 2.2)

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

add_executable(tests_cpp
    tests/main.cpp
    tests/Layout.cpp
    tests/ArrayImpl.cpp
)
target_link_libraries(tests_cpp Catch2::Catch2 ndarray)
include(Catch)
catch_discover_tests(tests_cpp)

if(pybind11_FOUND)

    function(ndarray_add_python_test_module MODULE_NAME)
        configure_file(tests/${MODULE_NAME}.py tests/${MODULE_NAME}.py COPYONLY)
    endfunction(ndarray_add_python_test_module)

    # TODO: use unittest discovery and TEST_INCLUDE_FILES to add finer-grained
    #       tests dynamically

    function(ndarray_add_python_test TEST_NAME)
        set(FULL_TEST_NAME ${TEST_NAME}_Python)
        ndarray_add_python_test_module(test_${TEST_NAME})
        add_test(
            NAME
                ${FULL_TEST_NAME}
            COMMAND
                ${PYTHON_EXECUTABLE} -m unittest tests.test_${TEST_NAME}
            WORKING_DIRECTORY
                ${CMAKE_CURRENT_BINARY_DIR}
        )
        set_tests_properties(
            ${FULL_TEST_NAME}
            PROPERTIES LABELS "python;${TEST_NAME}"
        )
    endfunction(ndarray_add_python_test)

    ndarray_add_python_test_module(__init__)
    ndarray_add_python_test_module(compilation)
    ndarray_add_python_test(Array)

else()

    message("pybind11 not found; not running Python tests.")

endif()
