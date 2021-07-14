include(llvm-virtual)

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_USE_LIBCXX ${USE_LIBCXX} CACHE BOOL "" FORCE)

include(external/CMakeLists.txt.googlebenchmark.in)

export(TARGETS benchmark FILE GoogleBenchmarkConfig.cmake)
