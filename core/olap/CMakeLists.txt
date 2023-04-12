project(olap VERSION 0.1 LANGUAGES CXX)

if (NOT USE_CUDA)
  message(FATAL_ERROR "Warning: Building without support for GPUs (nvcc not detected)")
endif ()

# Pure, regular C++ files
set(olap_cxx
  lib/common/common.cpp
  lib/common/llvm-dump.cpp
  lib/common/symantec-config.cpp
  lib/expressions/expressions/ref-expression.cpp
  lib/expressions/binary-operators.cpp
  lib/expressions/expressions-dot-evaluator.cpp
  lib/expressions/expressions-flusher.cpp
  lib/expressions/expressions-generator.cpp
  lib/expressions/expressions-hasher.cpp
  lib/expressions/expressions.cpp
  lib/expressions/path.cpp
  lib/operators/join/radix/radix-join-build.cpp
  lib/operators/cpu-to-gpu.cpp
  lib/operators/dict-scan.cpp
  lib/operators/flush.cpp
  lib/operators/bloom-filter/bloom-filter.cpp
  lib/operators/bloom-filter/bloom-filter-build.cpp
  lib/operators/bloom-filter/bloom-filter-probe.cpp
  lib/operators/bloom-filter/bloom-filter-repack.cpp
  lib/operators/gpu/gmonoids.cpp
  lib/operators/gpu/gpu-hash-group-by-chained.cpp
  lib/operators/gpu/gpu-hash-join-chained.cpp
  lib/operators/gpu/gpu-reduce.cpp
  lib/operators/gpu/gpu-sort.cpp
  lib/operators/gpu/gpu-to-cpu.cpp
  lib/operators/mem-move/mem-broadcast-device.cpp
  lib/operators/mem-move/mem-broadcast-scaleout.cpp
  lib/operators/mem-move/mem-move-device.cpp
  lib/operators/mem-move/mem-move-local-to.cpp
  lib/operators/mem-move/mem-move-scaleout.cpp
  lib/operators/router/router.cpp
  lib/operators/router/router-scaleout.cpp
  lib/operators/agg-t.cpp
  lib/operators/hash-group-by-chained.cpp
  lib/operators/hash-join-chained.cpp
  lib/operators/hash-join-chained-morsel.cpp
  lib/operators/hash-rearrange-buffered.cpp
  lib/operators/hash-rearrange.cpp
  lib/operators/join.cpp
  lib/operators/materializer-expr.cpp
  lib/operators/nest-opt.cpp
  lib/operators/nest.cpp
  lib/operators/null-filter.cpp
  lib/operators/outer-unnest.cpp
  lib/operators/packet-zip.cpp
  lib/operators/print.cpp
  lib/operators/project.cpp
  lib/operators/radix-join.cpp
  lib/operators/radix-nest.cpp
  lib/operators/reduce-opt.cpp
  lib/operators/relbuilder.cpp
  lib/operators/relbuilder-factory.cpp
  lib/operators/root.cpp
  lib/operators/scan.cpp
  lib/operators/select.cpp
  lib/operators/sort.cpp
  lib/operators/split.cpp
  lib/operators/unionall.cpp
  lib/operators/unnest.cpp
  lib/operators/block-to-tuples.cpp
  lib/operators/update.cpp
  lib/operators/gpu/gpu-hash-rearrange.cpp
  lib/operators/gpu/gpu-partitioned-hash-join-chained.cpp
  lib/plan/expression-parser.cpp
  lib/plan/plan-parser.cpp
  lib/plan/prepared-statement.cpp
  lib/plan/query-result.cpp
  lib/plugins/vector/vector-plugin.cpp
  lib/plugins/binary-block-plugin.cpp
  lib/plugins/binary-block-plugin-runtime-data-handles.cpp
  lib/plugins/binary-col-plugin.cpp
  lib/plugins/binary-internal-plugin.cpp
  lib/plugins/binary-row-plugin.cpp
  lib/plugins/csv-plugin-pm.cpp
  lib/plugins/csv-plugin.cpp
  lib/plugins/distributed-binary-block-plugin.cpp
  lib/plugins/json-jsmn-plugin.cpp
  lib/plugins/json-plugin.cpp
  lib/plugins/registry-based-binary-block-plugin.cpp
  lib/plugins/output/plugins-output.cpp
  lib/routing/affinitization-factory.cpp
  lib/routing/affinitizers.cpp
  lib/routing/routing-policy.cpp
  lib/util/atois.cpp
  lib/util/caching.cpp
  lib/util/catalog.cpp
  lib/util/context.cpp
  lib/util/demangle.cpp
  lib/util/flush-operator-tree.cpp
  lib/util/functions.cpp
  lib/util/gpu/gpu-intrinsics.cpp
  lib/util/jit/control-flow/if-statement.cpp
  lib/util/jit/cpu-module.cpp
  lib/util/jit/cpu-pipeline.cpp
  lib/util/jit/gpu-module.cpp
  lib/util/jit/gpu-pipeline.cpp
  lib/util/jit/jit-module.cpp
  lib/util/jit/pipeline.cpp
  lib/util/parallel-context.cpp
  lib/util/project-record.cpp
  lib/util/slack-limiter.cpp
  lib/values/expressionTypes.cpp
  lib/values/indexed-seq.cpp
  )

# C++ files requiring CUDA headers
set(olap_cxx_cu
  lib/operators/gpu/gpu-partitioned-hash-join-chained-helpers.cpp
  )

foreach (target ${olap_cu})
  set_source_files_properties(${target} PROPERTIES LANGUAGE CXX)
endforeach ()

foreach (target ${olap_cxx_cu} ${olap_cu})
  set_source_files_properties(${target} PROPERTIES COMPILE_FLAGS ${CUDA_CXXFLAGS})
endforeach ()

add_library(${PROJECT_NAME} SHARED
  ${olap_cxx_cu}
  ${olap_cxx}
  )

# If we have compiler requirements for this library, list them
# here
target_compile_features(${PROJECT_NAME}
  PUBLIC
  cxx_std_20
  )

# Depend on a library that we defined in the top-level file
target_link_libraries(${PROJECT_NAME}
  PUBLIC
  RapidJSON::RapidJSON
  LLVM::LLVM
  proteus::platform
  proteus::storage
  jsmn::jsmn
  rt # for shared memory
  )

include(proteus-lib)
add_library(proteus::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

# Unit-tests
if (UNIT_TESTS)
  add_subdirectory(tests)
  add_subdirectory(perftests)
endif ()
