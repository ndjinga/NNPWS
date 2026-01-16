set(MY_TEST_SOURCES NNPWS_tests.cxx)

# build test main exe
add_executable(NNPWS_test ${MY_TEST_SOURCES})
target_link_libraries(NNPWS_test PRIVATE NNPWSLib)

# Region 1
add_test(NAME Test_Region1_Liquide COMMAND NNPWS_test Region1)

# Region 2
add_test(NAME Test_Region2_Vapeur COMMAND NNPWS_test Region2)

# Test Batch
add_test(NAME Test_Systeme_Global COMMAND NNPWS_test Systeme)

add_test(NAME Test_Global_Performance COMMAND NNPWS_test Performance)

# lance tout
# add_test(NAME Test_Complet_All COMMAND NNPWS_test)
#install(TARGETS NNPWS_test DESTINATION test)


if(  NNPWS_WITH_PYTHON )
  add_subdirectory(python)
endif(NNPWS_WITH_PYTHON )

