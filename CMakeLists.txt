cmake_minimum_required(VERSION 2.8.11)

project(ndarray)

# Build is split into two broad passes:
# (1) Generation, which performs macro-expansion and header installation.
# (2) Test, which builds & executes zero-dependency core and (optional) dependent tests.
#
# The generation phase does not rely on the presence of external libraries,
# though the installed headers may be dependent on these libraries. This allows
# generation & installation of ndarray as a vendorized dependency of external
# projects without ndarry's optional dependencies.
#
# The test phase may rely on external libraries, which are enabled via the
# options below. Library resolution is deferred to tests/CMakeLists.txt and
# only executed if NDARRAY_TEST is enabled.
option(NDARRAY_TEST "Enable tests?" ON)
option(NDARRAY_EIGEN "Enable Eigen tests?" ON)
option(NDARRAY_FFTW "Enable FFTW tests?" ON)
option(NDARRAY_PYBIND11 "Enable Pybind11 tests?" OFF)

# enable C++11 support
add_definitions(-std=c++11)

# put our local cmake find scripts at the beginning of the cmake
# module search path
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})


# add a target to generate API documentation with Doxygen
find_package(Doxygen)
if(DOXYGEN_FOUND)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/doxygen.conf.in
                   ${CMAKE_CURRENT_BINARY_DIR}/doxygen.conf @ONLY)
    add_custom_target(doc ${DOXYGEN_EXECUTABLE}
                      ${CMAKE_CURRENT_BINARY_DIR}/doxygen.conf
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running Doxygen" VERBATIM
)
endif(DOXYGEN_FOUND)

if(NDARRAY_TEST)
    enable_testing()
    add_subdirectory(tests)
endif(NDARRAY_TEST)

# installation
install(DIRECTORY include/ DESTINATION include/
        FILES_MATCHING PATTERN "*.h")
