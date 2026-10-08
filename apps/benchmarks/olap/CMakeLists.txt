add_subdirectory(prepared-queries/ssb)
add_subdirectory(prepared-queries/tpch)
add_subdirectory(prepared-queries/stack)

# Benchmarks
add_subdirectory(benchmark-ssb)
add_subdirectory(benchmark-tpch)
add_subdirectory(benchmark-stack)

if (NOT PROTEUS_CB2T_ARTIFACT)
  add_subdirectory(prepared-queries/ssb100_bloom)
  add_subdirectory(prepared-queries/ch100w)
endif()
