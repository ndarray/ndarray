cmake_minimum_required(VERSION 3.15)
project(ndarray LANGUAGES CXX)
set (CMAKE_CXX_STANDARD 11)

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

add_subdirectory(include)

find_package(Doxygen)
if(DOXYGEN_FOUND)
    add_subdirectory(doc)  
endif(DOXYGEN_FOUND)

if(NDARRAY_TEST)
    enable_testing()
    add_subdirectory(tests)
endif(NDARRAY_TEST)

