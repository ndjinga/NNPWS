INCLUDE(${SWIG_USE_FILE})

######################################
# setup has been done in project level, just do it again
######################################

# cross OS setup
IF(WIN32)
   # windows
   SET(PY_EXTENSION "pyd")
ELSE(WIN32)
   SET(PY_EXTENSION "so")
ENDIF(WIN32)

# DO NOT use relative include path within *.i file, but add the include_path
include_directories(${MY_HEADER_DIR} ${PYTHON_INCLUDE_DIRS})

set(MY_MODULE "NNPWS")  # it is possible to use the same name as sharedlib name
set(MY_SHARED_LIB "NNPWS")
set(MY_SWIG_INTERFACE "NNPWS.i")

######################################
# swig build
######################################
file(COPY NNPWS.i DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
SET_SOURCE_FILES_PROPERTIES(${CMAKE_CURRENT_BINARY_DIR}/NNPWS.i PROPERTIES SWIG_DEFINITIONS "-shadow")

if(${CMAKE_VERSION} VERSION_GREATER "3.8.0") 
  SWIG_ADD_LIBRARY(NNPWS TYPE SHARED LANGUAGE python SOURCES ${CMAKE_CURRENT_BINARY_DIR}/NNPWS.i)
else(${CMAKE_VERSION} VERSION_GREATER "3.8.0")  # this section is not tested
  swig_add_module(${MY_MODULE}   python    ${MY_SWIG_INTERFACE}   )
endif(${CMAKE_VERSION} VERSION_GREATER "3.8.0")

SWIG_LINK_LIBRARIES(NNPWS ${PYTHON_LIBRARIES}  NNPWSLib ${GSL_LIBRARIES} )

INSTALL(TARGETS _NNPWS DESTINATION lib/NNPWS)
INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/NNPWS.py DESTINATION bin/NNPWS)

file(COPY ${PROJECT_SOURCE_DIR}/python/ DESTINATION ${CMAKE_CURRENT_BINARY_DIR} FILES_MATCHING PATTERN "*.py")
