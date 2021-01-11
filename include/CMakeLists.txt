add_library(${PROJECT_NAME} INTERFACE)

target_include_directories(
    ${PROJECT_NAME} INTERFACE $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
# add a target to generate API documentation with Doxygen

# installation
include(GNUInstallDirs)

install(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}-config)

install(
    EXPORT ${PROJECT_NAME}-config 
    NAMESPACE ${PROJECT_NAME}::                                                         
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/cmake/${PROJECT_NAME})

install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/${PROJECT_NAME}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} )

