function(add_proteus_perftest_nolib target)
  add_executable(${target} ${ARGN})

  target_link_libraries(${target} benchmark::benchmark_main)
  target_enable_default_warnings(${target})

  set(_proteus_install_target ${target})
  set(_proteus_install_dev ${PROTEUS_CPACK_COMP_DEV})
  set(_proteus_install_bin ${PROTEUS_CPACK_COMP_TESTS})
  include(_proteus-install)

  add_test(NAME PT-${target}
    COMMAND ${target}
    WORKING_DIRECTORY ${CMAKE_INSTALL_BINDIR}
    )
endfunction()

function(add_proteus_perftest_with_custom_main_nolib target)
  add_executable(${target} ${ARGN})
  # If we include benchmark::benchmark,
  # we should use this to avoid warnings due to benchmark's headers
  function(target_link_libraries_system target)
    # from https://stackoverflow.com/a/52136420/1237824
    set(libs ${ARGN})
    foreach(lib ${libs})
      get_target_property(lib_include_dirs ${lib} INTERFACE_INCLUDE_DIRECTORIES)
      target_include_directories(${target} SYSTEM PRIVATE ${lib_include_dirs})
      target_link_libraries(${target} ${lib})
    endforeach(lib)
  endfunction(target_link_libraries_system)

  target_link_libraries_system(${target} benchmark::benchmark)
  target_enable_default_warnings(${target})

  set(_proteus_install_target ${target})
  set(_proteus_install_dev ${PROTEUS_CPACK_COMP_DEV})
  set(_proteus_install_bin ${PROTEUS_CPACK_COMP_TESTS})
  include(_proteus-install)

  add_test(NAME PT-${target}
    COMMAND ${target}
    WORKING_DIRECTORY ${CMAKE_INSTALL_BINDIR}
    )
endfunction()

function(add_proteus_perftest target tested_lib)
  add_proteus_perftest_nolib(${target} ${ARGN})
  target_link_libraries(${target} ${tested_lib})
  target_include_directories(${target} PRIVATE $<TARGET_PROPERTY:${tested_lib},INCLUDE_DIRECTORIES>)
endfunction()

function(add_proteus_perftest_with_custom_main target tested_lib)
  add_proteus_perftest_with_custom_main_nolib(${target} ${ARGN})
  target_link_libraries(${target} ${tested_lib})
  target_include_directories(${target} PRIVATE $<TARGET_PROPERTY:${tested_lib},INCLUDE_DIRECTORIES>)
endfunction()
