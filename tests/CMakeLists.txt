#-----------------------------------------------------------------------
# Copy data files too in the destination folder
#-----------------------------------------------------------------------
# Creates <..>/pelago/inputs
install(DIRECTORY "inputs"
  DESTINATION ${CMAKE_INSTALL_BINDIR}
  #COMPONENT ${PROJECT_NAME}_${PROTEUS_CPACK_COMP_SUFFIX_TESTS}
  COMPONENT ${PROTEUS_CPACK_COMP_TESTS}
  )
# Creates <..>/pelago/outputs
install(DIRECTORY "olap/outputs"
  DESTINATION ${CMAKE_INSTALL_BINDIR}
  #COMPONENT ${PROJECT_NAME}_${PROTEUS_CPACK_COMP_SUFFIX_TESTS}
  COMPONENT ${PROTEUS_CPACK_COMP_TESTS}
  )

#-----------------------------------------------------------------------
# Define path to installed resources
#-----------------------------------------------------------------------
add_definitions(-DTEST_INPUTS="inputs")
add_definitions(-DTEST_OUTPUTS="outputs")

# Unit-tests
if (UNIT_TESTS)
  add_subdirectory(olap)
  add_subdirectory(oltp)
endif ()

# Deprecated unit-tests
if (DUNIT_TESTS)
  add_subdirectory(olap/deprecated)
endif ()
