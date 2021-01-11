configure_file(${CMAKE_CURRENT_SOURCE_DIR}/doxygen.conf.in
               ${CMAKE_CURRENT_BINARY_DIR}/doxygen.conf @ONLY)
add_custom_target(doc ${DOXYGEN_EXECUTABLE}
                  ${CMAKE_CURRENT_BINARY_DIR}/doxygen.conf
                  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                  COMMENT "Running Doxygen" VERBATIM)

