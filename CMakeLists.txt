cmake_minimum_required(VERSION 3.12.0)

project("NNPWS" LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(PROJECT_BRIEF "A library providing IAPWS-97 functions based on neural networks")

option (NNPWS_WITH_PYTHON     "Build Python interface of NNPWS" ON)
option (NNPWS_WITH_TESTS      "Build and run NNPWS example tests" ON)
option (NNPWS_WITH_DOCUMENTATION "Generate NNPWS documentation" ON)

#Path to installed libraries
set (PYTHON_ROOT_DIR       OFF CACHE STRING "Python  library path" )
set (PyTORCH_ROOT_DIR      OFF CACHE STRING "Libtorch library path" )

list (APPEND CMAKE_MODULE_PATH "${NNPWS_SOURCE_DIR}/cmake" "${PyTORCH_ROOT_DIR}/share/cmake")

set(MY_SOURCES
src/NNPWS.cxx
src/ModelLoader.cxx
src/FastInference.cxx
)
set(MY_HEADERS
src
)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_library(NNPWSLib SHARED ${MY_SOURCES})
target_link_libraries(NNPWSLib "${TORCH_LIBRARIES}")
target_include_directories(NNPWSLib PRIVATE ${MY_HEADERS} ${TORCH_INCLUDE_DIRS})
install(TARGETS NNPWSLib DESTINATION lib)
install(FILES ${PROJECT_SOURCE_DIR}/src/NNPWS.hxx DESTINATION include) #

#add_executable(MainTest src/main.cxx)
#target_link_libraries(MainTest NNPWSLib)

if( NNPWS_WITH_PYTHON )
  message(STATUS "NNPWS requested with PYTHON library. Searching PYTHON library.")
  find_package(Python REQUIRED COMPONENTS Interpreter Development )
  find_package (SWIG 3.0 REQUIRED)
  add_subdirectory(swig)
 set (CPACK_DEBIAN_PACKAGE_DEPENDS "${CPACK_DEBIAN_PACKAGE_DEPENDS}, python3, swig, python3-matplotlib")
 set (CPACK_RPM_PACKAGE_REQUIRES   "${CPACK_RPM_PACKAGE_REQUIRES},   python3, swig, python3-matplotlib")
endif( NNPWS_WITH_PYTHON )

if( NNPWS_WITH_TESTS )
 enable_testing ()
 add_subdirectory(tests)
endif( NNPWS_WITH_TESTS )

# Find and use Doxygen for documentation
if (NNPWS_WITH_DOCUMENTATION)                                                                           #
 IF   ( DEFINED DOXYGEN_ROOT_DIR OR DEFINED ENV{DOXYGEN_ROOT_DIR})
   IF   ( DEFINED DOXYGEN_ROOT_DIR )
     set(DOXYGEN_EXECUTABLE    ${DOXYGEN_ROOT_DIR}/bin/doxygen )
   ELSE ( DEFINED DOXYGEN_ROOT_DIR )
     set(DOXYGEN_EXECUTABLE $ENV{DOXYGEN_ROOT_DIR}/bin/doxygen )
   ENDIF( DEFINED DOXYGEN_ROOT_DIR)
 ENDIF( DEFINED DOXYGEN_ROOT_DIR OR DEFINED ENV{DOXYGEN_ROOT_DIR})
  find_package (Doxygen)                                                                                    #
  if   (NOT DOXYGEN_FOUND)
    message (FATAL_ERROR "Doxygen is needed to build the documentation. Please install it correctly.")
  endif(NOT DOXYGEN_FOUND)

  set (CPACK_DEBIAN_PACKAGE_DEPENDS "${CPACK_DEBIAN_PACKAGE_DEPENDS}, doxygen, graphviz, mscgen")
  set (CPACK_RPM_PACKAGE_REQUIRES   "${CPACK_RPM_PACKAGE_REQUIRES},   doxygen, graphviz, mscgen")

  configure_file (doc/Doxyfile-NNPWS.in ${PROJECT_BINARY_DIR}/Doxyfile-dev-doc  @ONLY IMMEDIATE)
  add_custom_target (doc COMMAND ${DOXYGEN_EXECUTABLE} ${PROJECT_BINARY_DIR}/Doxyfile-dev-doc  SOURCES ${PROJECT_BINARY_DIR}/Doxyfile-dev-doc)			                                            #
  install (DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/doc/dev-doc/html DESTINATION share/doc/NNPWS-dev-doc OPTIONAL)                  #

endif(NNPWS_WITH_DOCUMENTATION)

