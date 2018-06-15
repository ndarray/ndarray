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

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${Boost_INCLUDE_DIRS}
)

add_executable(ndarray ndarray.cc)
target_link_libraries(ndarray ${Boost_LIBRARIES})
add_test(test_ndarray ndarray)

add_executable(views views.cc)
target_link_libraries(views ${Boost_LIBRARIES})
add_test(test_views views)


### Eigen dependency tests
if(NDARRAY_EIGEN)
    find_package(Eigen3 REQUIRED)

    include_directories( ${EIGEN3_INCLUDE_DIR} )

    add_executable(ndarray-eigen ndarray-eigen.cc)
    target_link_libraries(ndarray-eigen ${Boost_LIBRARIES})
    add_test(test_ndarray_eigen ndarray-eigen)
endif(NDARRAY_EIGEN)

### FFTW dependency tests
if(NDARRAY_FFTW)
    find_package(FFTW REQUIRED)
    include_directories( ${FFTW_INCLUDES} )

    add_executable(ndarray-fft ndarray-fft.cc)
    target_link_libraries(ndarray-fft ${Boost_LIBRARIES} ${FFTW_LIBRARIES})
    add_test(test_ndarray-fft ndarray-fft)
endif(NDARRAY_FFTW)

### Python resolution
if(NDARRAY_PYBIND11)
    find_package(PythonInterp REQUIRED)
    find_package(PythonLibs REQUIRED)
    find_package(NumPy REQUIRED)

    include_directories(
        ${PYTHON_INCLUDE_DIRS}
        ${NUMPY_INCLUDE_DIRS}
    )
ENDIF()

###Pybind11 dependency tests (also depend on Eigen)
if(NDARRAY_PYBIND11)
    if(NDARRAY_EIGEN)
        find_package(pybind11 REQUIRED)

        include_directories(${PYBIND11_INCLUDE_DIR})

        # Either pybind11 2.1.x doesn't set the module extension appropriately,
        # or ndarray is inadvertently squashing it.  In any case, the problem
        # goes away with pybind11 2.2.x, and in the meantime this fixes it.
        # It probably wouldn't work on Windows, but we don't care about that.
        set(PYTHON_MODULE_EXTENSION ".so")

        pybind11_add_module(pybind11_test_mod pybind11_test_mod.cc)

        configure_file(pybind11_test.py pybind11_test.py COPYONLY)
        add_test(NAME pybind11_test
            COMMAND ${PYTHON_EXECUTABLE}
            ${CMAKE_CURRENT_BINARY_DIR}/pybind11_test.py)
    else(NDARRAY_EIGEN)
        message(STATUS "Skipping pybind11 tests as they depend on Eigen")
    endif(NDARRAY_EIGEN)
endif(NDARRAY_PYBIND11)
