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
install(FILES ${MY_HEADERS} DESTINATION include)


if( NNPWS_WITH_PYTHON )
  message(STATUS "NNPWS requested with PYTHON library. Searching PYTHON library.")
  find_package(Python REQUIRED COMPONENTS Interpreter Development )
  find_package (SWIG 3.0 REQUIRED)
  add_subdirectory(swig)
endif( NNPWS_WITH_PYTHON )

if( NNPWS_WITH_TESTS )
 enable_testing ()

 #set(MY_TEST_SOURCES
 #tests/src/testNNPWS_nxn_par.c
 #tests/src/testNNPWS_2x2_Shat.c
 #)

 # build test main exe
 # add_executable(testNNPWS_initial                      tests/src/testNNPWS_initial.c)
 # target_include_directories(testNNPWS_initial                      PRIVATE ${PETSC_INCLUDES} )
 # target_link_libraries(testNNPWS_initial                      ${PETSC_LIBRARIES} ${MPI_HOME}/lib/libmpi.so)

 # add_test(NAME testNNPWS_initial_Assembly_1                 COMMAND "./testNNPWS_initial"                 "-f0" "${PROJECT_SOURCE_DIR}/tests/data/matrices/3DAssemblyMesh/Assembly_PolyMAC_1_vp.petsc" "-nU" "20344" "-nP" "9552" "-pc_fieldsplit_type" "schur" "-pc_fieldsplit_schur_fact_type" "lower" "-fieldsplit_0_ksp_type" "gmres" "-fieldsplit_0_pc_type" "gamg" "-fieldsplit_0_pc_gamg_type" "agg" "-fieldsplit_1_ksp_type" "preonly" "-fieldsplit_1_pc_type" "lu")
 # install(TARGETS testNNPWS_initial DESTINATION tests)

 add_subdirectory(tests)
endif( NNPWS_WITH_TESTS )

