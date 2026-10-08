if (PROTEUS_CB2T_ARTIFACT)
  add_subdirectory(standalones/olap/flags)
  add_subdirectory(benchmarks/olap)
  add_subdirectory(standalones/adm/adm-partition-tpch)
else()
  add_subdirectory(standalones)

  # Benchmarks
  if (NOT PLATFORM_ONLY)
    add_subdirectory(benchmarks/olap)
    add_subdirectory(benchmarks/oltp)
    add_subdirectory(benchmarks/htap)
  endif()
endif()
