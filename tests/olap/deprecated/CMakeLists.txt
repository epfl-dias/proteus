if (FALSE)
  set(TS_NAME unit-tests-olap-deprecated)

  add_executable(${TS_NAME}
    tests-hashing.cpp
    tests-json.cpp
    tests-nest.cpp
    tests-operators.cpp
    tests-output.cpp
    tests-sailors.cpp
    tests-string.cpp
    )

  target_link_libraries(${TS_NAME}
    proteus::olap
    )

else ()

  set(TS_TESTS
    tests-csv
    tests-hashing
    tests-json
    tests-nest
    tests-operators
    tests-output
    tests-sailors
    )

  set(TS_COMMON_SRCS
    )

  set(TS_COMMON_LIBS
    proteus::olap
    )
endif ()

include(proteus-test)
