cmake_minimum_required(VERSION 2.8.11)

project(ndarray)

option(NDARRAY_TEST "Enable tests?" ON)
option(NDARRAY_SWIG "Enable Swig?" ON)
option(NDARRAY_BOOST_PYTHON "Enable Boost Python?" OFF)
option(NDARRAY_PYBIND11 "Enable Pybind11?" OFF)

# enable C++11 support
add_definitions(-std=c++11)

# put our local cmake find scripts at the beginning of the cmake
# module search path
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

# find required packages
find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(NumPy REQUIRED)
find_package(Boost COMPONENTS unit_test_framework REQUIRED)
find_package(FFTW REQUIRED)

# allow user to specify EIGEN_DIR as an environment variable
# this is necessary to integrate with build systems that install Eigen
# in non-standard locations
set(EIGEN3_INCLUDE_DIR $ENV{EIGEN_DIR})
if(NOT EIGEN3_INCLUDE_DIR)
	find_package(Eigen3 REQUIRED)
else()
	set(EIGEN3_INCLUDE_DIR ${EIGEN3_INCLUDE_DIR}/include)
endif(NOT EIGEN3_INCLUDE_DIR)

if (NDARRAY_SWIG)
	find_package(SWIG REQUIRED)
endif (NDARRAY_SWIG)

if (NDARRAY_PYBIND11)
	find_package(Pybind11 REQUIRED)
endif (NDARRAY_PYBIND11)

include_directories(
	${PROJECT_SOURCE_DIR}/include
	${PYTHON_INCLUDE_DIRS}
	${NUMPY_INCLUDE_DIRS}
	${Boost_INCLUDE_DIRS}
	${FFTW_INCLUDES}
	${EIGEN3_INCLUDE_DIR}
	${PYBIND11_INCLUDE_DIR}
	)

add_subdirectory(include)

if(NDARRAY_TEST)
	enable_testing()
	add_subdirectory(tests)
endif(NDARRAY_TEST)

# installation
install(DIRECTORY include/ DESTINATION include/
	FILES_MATCHING PATTERN "*.h")

if(NDARRAY_SWIG)
	install(DIRECTORY include/ DESTINATION include/
		FILES_MATCHING PATTERN "*.i")
endif(NDARRAY_SWIG)
