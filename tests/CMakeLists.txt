set(MY_TEST_SOURCES ./NNPWS_tests.c )

# build test main exe
add_executable(NNPWS_test ${MY_TEST_SOURCES})
target_link_libraries(NNPWS_test NNPWSLib )

add_test(NAME NNPWS_test  COMMAND "./NNPWS_test")  
install(TARGETS NNPWS_test DESTINATION test)


if(   NNPWS_WITH_PYTHON )
  add_subdirectory(python)
endif(NNPWS_WITH_PYTHON )

