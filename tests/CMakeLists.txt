# Tests are split by feature & dependency and are enabled on a
# per-dependency basis.
#
# A test is formatted as:
# (1) An if-guard over an appropriate option.
# (2) Resolution and inclusion of any required dependencies.
# (3) Build of the test executable.
# (4) Addition of the test executable via add_test.

### Core tests, which rely only on boost-test and ndarray.
find_package(Boost COMPONENTS unit_test_framework REQUIRED)

include_directories( ${PROJECT_SOURCE_DIR}/include)

add_executable(ndarray_test ndarray.cc)

target_link_libraries(ndarray_test ndarray Boost::unit_test_framework)
add_test(test_ndarray ndarray_test)

add_executable(views views.cc)

target_link_libraries(views Boost::unit_test_framework )
add_test(test_views views)


### Eigen dependency tests
if(NDARRAY_EIGEN)
    find_package(Eigen3 REQUIRED)
    add_executable(ndarray-eigen ndarray-eigen.cc)
    target_link_libraries(ndarray-eigen ndarray Eigen3::Eigen Boost::unit_test_framework)
    add_test(test_ndarray_eigen ndarray-eigen)
endif(NDARRAY_EIGEN)

# note: this is a workaround for 
# https://github.com/FFTW/fftw3/issues/130
# one would prefer to use find_package(fftw3)
### FFTW dependency tests
if(NDARRAY_FFTW)
    include(FindPkgConfig)
    pkg_search_module(FFTW3 REQUIRED fftw3)
    include_directories( ${FFTW3_INCLUDES} )
    add_executable(ndarray-fft ndarray-fft.cc)
    target_link_libraries(ndarray-fft ndarray Boost::unit_test_framework ${FFTW3_LIBRARIES})
    add_test(test_ndarray-fft ndarray-fft)
endif(NDARRAY_FFTW)

### Python resolution
if(NDARRAY_PYBIND11)
    find_package(Python ${PYTHON_VERSION} COMPONENTS Interpreter Development NumPy REQUIRED)
ENDIF()

###Pybind11 dependency tests (also depend on Eigen)
if(NDARRAY_PYBIND11)
    if(NDARRAY_EIGEN)
        find_package(pybind11 REQUIRED)

        pybind11_add_module(pybind11_test_mod pybind11_test_mod.cc )
        target_link_libraries(pybind11_test_mod PRIVATE Eigen3::Eigen)
        configure_file(pybind11_test.py pybind11_test.py COPYONLY)
        add_test(NAME pybind11_test
            COMMAND ${Python_EXECUTABLE}
            ${CMAKE_CURRENT_BINARY_DIR}/pybind11_test.py)
    else(NDARRAY_EIGEN)
        message(STATUS "Skipping pybind11 tests as they depend on Eigen")
    endif(NDARRAY_EIGEN)
endif(NDARRAY_PYBIND11)
