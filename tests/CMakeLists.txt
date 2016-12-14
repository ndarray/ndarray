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
if(NDARRAY_SWIG OR NDARRAY_PYBIND11)
  find_package(PythonInterp REQUIRED)
  find_package(PythonLibs REQUIRED)
  find_package(NumPy REQUIRED)

  include_directories(
    ${PYTHON_INCLUDE_DIRS}
    ${NUMPY_INCLUDE_DIRS}
  )
ENDIF()

###SWIG dependency tests
if(NDARRAY_SWIG)
	find_package(SWIG REQUIRED)
	include(${SWIG_USE_FILE})

	set_property(SOURCE swig_test_mod.i PROPERTY CPLUSPLUS ON)
	swig_add_module(swig_test_mod python swig_test_mod.i)

	if (APPLE)
		swig_link_libraries(swig_test_mod "-undefined dynamic_lookup")
	else ()
		swig_link_libraries(swig_test_mod "-shared")
	endif()

	configure_file(swig_test.py swig_test.py COPYONLY)
	add_test(NAME swig_test
		COMMAND ${PYTHON_EXECUTABLE}
		${CMAKE_CURRENT_BINARY_DIR}/swig_test.py)
endif(NDARRAY_SWIG)

###Pybind11 dependency tests
if(NDARRAY_PYBIND11)
	find_package(Pybind11 REQUIRED)

  include_directories(${PYBIND11_INCLUDE_DIR})

	add_library(pybind11_test_mod.so MODULE pybind11_test_mod.cc)
	set_target_properties(pybind11_test_mod.so PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}")
	set_target_properties(pybind11_test_mod.so PROPERTIES SUFFIX "${PYTHON_MODULE_EXTENSION}")

	if (APPLE)
		target_link_libraries(pybind11_test_mod.so PRIVATE "-undefined dynamic_lookup")
	else ()
		target_link_libraries(pybind11_test_mod.so "-shared")
	endif()

	configure_file(pybind11_test.py pybind11_test.py COPYONLY)
	add_test(NAME pybind11_test
		COMMAND ${PYTHON_EXECUTABLE}
		${CMAKE_CURRENT_BINARY_DIR}/pybind11_test.py)
endif(NDARRAY_PYBIND11)
