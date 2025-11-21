cmake_minimum_required(VERSION 3.12.0)

project("NNPWS")
set(PROJECT_BRIEF "A library providing IAPWS-97 functions based on neural networks")

list (APPEND CMAKE_MODULE_PATH "${NNPWS_SOURCE_DIR}/cmake")

option (NNPWS_WITH_PYTHON     "Build Python interface of NNPWS" ON)
option (NNPWS_WITH_TESTS      "Build and run NNPWS example tests" ON)
option (NNPWS_WITH_DOCUMENTATION "Generate NNPWS documentation" ON)

#Path to installed libraries
set (PYTHON_ROOT_DIR      OFF CACHE STRING "Python library path" )

set(MY_SOURCES
src/NNPWS.cxx
)
set(MY_HEADERS
src
)

add_library(NNPWSLib SHARED ${MY_SOURCES})
target_include_directories(NNPWSLib PRIVATE ${MY_HEADERS})
install(TARGETS NNPWSLib DESTINATION lib)
install(FILES PROJECT_SOURCE_DIR/src/NNPWS.hxx DESTINATION include) #


if( NNPWS_WITH_PYTHON )
  message(STATUS "NNPWS requested with PYTHON library. Searching PYTHON library.")
  find_package(Python REQUIRED COMPONENTS Interpreter Development )
  find_package (SWIG 3.0 REQUIRED)
  add_subdirectory(swig)
endif( NNPWS_WITH_PYTHON )

if( NNPWS_WITH_TESTS )
 enable_testing ()
 add_subdirectory(tests)
endif( NNPWS_WITH_TESTS )

