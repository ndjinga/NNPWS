INCLUDE(${SWIG_USE_FILE})

SET_SOURCE_FILES_PROPERTIES(NNPWS.i PROPERTIES CPLUSPLUS ON)
SET_SOURCE_FILES_PROPERTIES(NNPWS.i PROPERTIES C ON)
SET_SOURCE_FILES_PROPERTIES(NNPWS.i PROPERTIES SWIG_DEFINITIONS "-shadow")

INCLUDE_DIRECTORIES(
  ${Python_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/src							    #
)


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

######################################
# swig build
######################################
file(COPY NNPWS.i DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

if(${CMAKE_VERSION} VERSION_GREATER "3.8.0") 
  SWIG_ADD_LIBRARY(NNPWS TYPE SHARED LANGUAGE python SOURCES ${CMAKE_CURRENT_BINARY_DIR}/NNPWS.i)
else(${CMAKE_VERSION} VERSION_GREATER "3.8.0")  # this section is not tested
  swig_add_module(${MY_MODULE}   python    ${MY_SWIG_INTERFACE}   )
endif(${CMAKE_VERSION} VERSION_GREATER "3.8.0")

set_target_properties(_NNPWS PROPERTIES LINKER_LANGUAGE CXX)
SWIG_LINK_LIBRARIES(NNPWS  NNPWSLib )
#SWIG_LINK_LIBRARIES(NNPWS PUBLIC NNPWSLib )

INSTALL(TARGETS _NNPWS DESTINATION lib/NNPWS)
INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/NNPWS.py DESTINATION bin/NNPWS)

file(COPY ${PROJECT_SOURCE_DIR}/python/ DESTINATION ${CMAKE_CURRENT_BINARY_DIR} FILES_MATCHING PATTERN "*.py")
