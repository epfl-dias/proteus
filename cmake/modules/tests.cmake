function(add_proteus_unit_test_nolib target)
  add_executable(${target} ${ARGN})
  target_link_libraries(${target} ${GTEST_MAIN} ${GTEST})
  target_enable_default_warnings(${target})

  set(_proteus_install_target ${target})
  set(_proteus_install_dev ${PROTEUS_CPACK_COMP_DEV})
  set(_proteus_install_bin ${PROTEUS_CPACK_COMP_TESTS})
  include(_proteus-install)

  add_test(NAME UT-${target}
    COMMAND ${target}
    WORKING_DIRECTORY ${CMAKE_INSTALL_BINDIR}
    )
endfunction()

function(add_proteus_unit_test target testedlib)
  add_proteus_unit_test_nolib(${target} ${ARGN})
  target_link_libraries(${target} ${testedlib})
  target_include_directories(${target} PRIVATE $<TARGET_PROPERTY:${testedlib},INCLUDE_DIRECTORIES>)
endfunction()
