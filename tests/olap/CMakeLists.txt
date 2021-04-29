if (FALSE)
  set(TS_NAME unit-tests-olap)

  add_executable(${TS_NAME}
    #	tests-joins.cpp
    #	tests-json.cpp
    tests-cow.cpp
    tests-cpu-ssb.cpp
    tests-gpu-ssb.cpp
    tests-hyb-ssb.cpp
    tests-output.cpp
    tests-plan-parsing.cpp
    tests-threads.cpp
    test-utils.cpp
    )

  target_link_libraries(${TS_NAME}
    proteus::olap
    )
else ()

  set(TS_TESTS
    #	tests-joins
    #	tests-json
    tests-cow
    tests-cpu-ssb
    tests-gpu-ssb
    tests-hyb-ssb
    tests-output
    tests-plan-parsing
    tests-threads
    )

  set(TS_COMMON_SRCS
    test-utils.cpp
    )

  set(TS_COMMON_LIBS
    proteus::olap
    )
endif ()

include(proteus-test)
