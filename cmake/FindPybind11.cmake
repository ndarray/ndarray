#.rst:
# FindPybind11
# --------
#
# Find the Pybind11 includes.
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# This module will set the following variables in your project::
#
#  PYBIND11_FOUND          - True if Pybind11 found on the local system
#  PYBIND11_INCLUDE_DIR    - Location of Pybind11 header files.
#
# Hints
# ^^^^^
#
# Set ``PYBIND11_DIR`` to a directory that contains a Pybind11 installation.
#
# This script expects to find headers at ``$PYBIND11_DIR/include``.
#

# Include these modules to handle the QUIETLY and REQUIRED arguments.
include(FindPackageHandleStandardArgs)

#=============================================================================
# If the user has provided ``PYBIND11_DIR``, use it!  Choose items found
# at this location over system locations.
if( EXISTS "$ENV{PYBIND11_DIR}" )
  file( TO_CMAKE_PATH "$ENV{PYBIND11_DIR}" PYBIND11_DIR )
  set( PYBIND11_DIR "${PYBIND11_DIR}" CACHE PATH "Prefix for Pybind11 installation." )
endif()

#=============================================================================
# Set PYBIND11_INCLUDE_DIRS.
# Try to find pybind11 at $PYBIND11_DIR (if provided) or in standard
# system locations.
find_path(PYBIND11_INCLUDE_DIR
	NAMES pybind11/pybind11.h
	HINTS ${PYBIND11_DIR}/include
	)

find_package_handle_standard_args(Pybind11 DEFAULT_MSG PYBIND11_INCLUDE_DIR)
