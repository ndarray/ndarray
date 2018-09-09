cmake_minimum_required(VERSION 3.5)
project(ndarray VERSION 2.0.0 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)  # used to obtain build commands for expected compile failure tests

option(TEST_THREADS "Build and run tests with threads enabled" ON)
option(TEST_CXX17 "Build and run tests with C++17 features" ON)

find_package(fmt 4.1 REQUIRED)
find_package(Catch2 2.2.3 REQUIRED)
find_package(pybind11 2.2)
find_package(Doxygen 1.8)

if(DOXYGEN_FOUND)
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/doc/doxygen.conf.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/doxygen.conf)
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
    add_custom_target(doc
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM
    )
else(DOXYGEN_FOUND)
    message("doc target disabled; Doxygen not found.")
endif(DOXYGEN_FOUND)

add_library(ndarray INTERFACE)
target_include_directories(ndarray
    INTERFACE
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(ndarray INTERFACE fmt::fmt)
if(TEST_CXX17)
    target_compile_features(ndarray
        INTERFACE
            cxx_std_17
    )
else(TEST_CXX17)
    target_compile_features(ndarray
        INTERFACE
            cxx_std_14
    )
endif(TEST_CXX17)

enable_testing()

add_executable(tests_cpp
    tests/main.cpp
    tests/Layout.cpp
    tests/ArrayImpl.cpp
    tests/Array.cpp
    tests/views.cpp
    tests/expressions.cpp
)
target_link_libraries(tests_cpp Catch2::Catch2 ndarray)
include(Catch)
catch_discover_tests(tests_cpp)

if(TEST_THREADS)
    set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
    set(THREADS_PREFER_PTHREAD_FLAG TRUE)
    find_package(Threads REQUIRED)
    target_link_libraries(tests_cpp Threads::Threads)
    target_compile_definitions(tests_cpp PRIVATE NDARRAY_TEST_WITH_THREADS=1)
endif(TEST_THREADS)


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
