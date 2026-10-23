function(add_proteus_olap_perftest target)
  add_proteus_perftest(${target} ${ARGN})
  target_link_libraries(${target} PUBLIC proteus::olap-perf-main)
endfunction()

add_subdirectory(olap-perf-util)

if (NOT PROTEUS_CB2T_ARTIFACT)
  add_subdirectory(olap-perf-main)
  add_subdirectory(query-perf)
  add_subdirectory(queue-perf)
endif()
