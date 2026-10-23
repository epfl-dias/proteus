project(olap VERSION 0.1 LANGUAGES CXX)

if (NOT USE_CUDA)
  message(FATAL_ERROR "Warning: Building without support for GPUs (nvcc not detected)")
endif ()

# Pure, regular C++ files
set(olap_cxx
  lib/common/common.cpp
  lib/common/olap-flags.cpp
  lib/common/llvm-dump.cpp
  lib/common/symantec-config.cpp
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
  lib/operators/mem-move/compression.cpp
  lib/operators/mem-move/mem-broadcast-device.cpp
  lib/operators/mem-move/mem-broadcast-scaleout.cpp
  lib/operators/mem-move/mem-move-device.cpp
  lib/operators/mem-move/mem-move-local-to.cpp
  lib/operators/mem-move/mem-move-scaleout.cpp
  lib/operators/router/generalized-router.cpp
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
  lib/operators/decompression-flags.cpp
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
  lib/operators/pre-decompress.cpp
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
  lib/plugins/binary-block-nvme-plugin.cpp
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
  lib/util/flush-operator-tree.cpp
  lib/util/functions.cpp
  lib/util/jit/control-flow/if-statement.cpp
  lib/util/jit/olap-pipeline.cpp
  lib/util/parallel-context.cpp
  lib/util/project-record.cpp
  lib/util/radix/aggregations/radix-aggr.cpp
  lib/util/radix/joins/radix-join.cpp
  lib/util/slack-limiter.cpp
)

# C++ files requiring CUDA headers
set(olap_cxx_cu
  lib/operators/decompression/flbp-decompression.cpp
  lib/operators/gpu/gpu-partitioned-hash-join-chained-helpers.cpp
  lib/operators/decompression/rfor-decompression.cpp
  lib/operators/gpu/compressed-block-to-tuples.cpp
)


# Use CUDA path from CMake options (CUDA_TOOLKIT_ROOT_DIR)
if(CUDA_TOOLKIT_ROOT_DIR)
  set(CUDA_PATH ${CUDA_TOOLKIT_ROOT_DIR})
endif()

set(OLAP_CUDA_CXXFLAGS "${CUDA_CXXFLAGS}")
string(REPLACE "--std=c++17" "--std=c++20" OLAP_CUDA_CXXFLAGS "${OLAP_CUDA_CXXFLAGS}")
if (NOT OLAP_CUDA_CXXFLAGS MATCHES "_LIBCPP_ENABLE_CXX20_REMOVED_TYPE_TRAITS")
  string(STRIP "${OLAP_CUDA_CXXFLAGS} -D_LIBCPP_ENABLE_CXX20_REMOVED_TYPE_TRAITS" OLAP_CUDA_CXXFLAGS)
endif()

# Set compile flags for files that need CUDA headers
foreach (target ${olap_cxx_cu})
  set_source_files_properties(${target} PROPERTIES
    COMPILE_FLAGS "--cuda-path=${CUDA_TOOLKIT_ROOT_DIR} -I${CUDA_TOOLKIT_ROOT_DIR}/include ${OLAP_CUDA_CXXFLAGS}")
endforeach()

add_library(${PROJECT_NAME} SHARED
  ${olap_cxx_cu}
  ${olap_cxx}
)

target_compile_features(${PROJECT_NAME}
  PUBLIC
  cxx_std_20
)

set(OLAP_PRIVATE_LIBS magic_enum::magic_enum lz4_static)
find_package(nvcomp)
if (nvcomp_FOUND)
  list(APPEND OLAP_PRIVATE_LIBS nvcomp::nvcomp)
  target_compile_definitions(${PROJECT_NAME} PRIVATE HAVE_NVCOMP)
endif ()

find_package(GDS REQUIRED)

# Depend on a library that we defined in the top-level file
target_link_libraries(${PROJECT_NAME}
  PUBLIC
  RapidJSON::RapidJSON
  LLVM::LLVM
  proteus::platform
  proteus::storage
  proteus::codegen
  jsmn::jsmn
  rt # for shared memory
  stduuid::stduuid
  NVIDIA::GDS
  PRIVATE
  ${OLAP_PRIVATE_LIBS}
  magic_enum::magic_enum
)

include(proteus-lib)
add_library(proteus::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

# Unit-tests
if (UNIT_TESTS)
  add_subdirectory(tests)
endif ()
add_subdirectory(perftests)
