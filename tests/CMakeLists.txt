add_executable(ndarray ndarray.cc)
target_link_libraries(ndarray ${Boost_LIBRARIES})

add_executable(views views.cc)
target_link_libraries(views ${Boost_LIBRARIES})

add_executable(ndarray-eigen ndarray-eigen.cc)
target_link_libraries(ndarray-eigen ${Boost_LIBRARIES})

add_executable(ndarray-fft ndarray-fft.cc)
target_link_libraries(ndarray-fft ${Boost_LIBRARIES} ${FFTW_LIBRARIES})

add_test(test_ndarray ndarray)
add_test(test_views views)
add_test(test_ndarray-fft ndarray-fft)
add_test(test_ndarray_eigen ndarray-eigen)

if(NDARRAY_SWIG)
	find_package(SWIG)
	include(${SWIG_USE_FILE})

	set_source_files_properties(swig_test_mod.i PROPERTIES CPLUSPLUS ON)
	swig_add_module(swig_test_mod python swig_test_mod.i)
	swig_link_libraries(swig_test_mod ${PYTHON_LIBRARIES})

	find_package(PythonInterp REQUIRED)
	configure_file(swig_test.py swig_test.py COPYONLY)
	add_test(NAME swig_test
		COMMAND ${PYTHON_EXECUTABLE}
		${CMAKE_CURRENT_BINARY_DIR}/swig_test.py)
endif(NDARRAY_SWIG)
