set(TS_TESTS
  tests-cow
  tests-cpu-ssb
  tests-gpu-ssb
  tests-hyb-ssb
  tests-output
  tests-plan-parsing
  tests-threads
  )

set(TS_COMMON_LIBS
  proteus::olap
  proteus::olap-test
  proteus::olap-test-utils
  )


include(proteus-test)
