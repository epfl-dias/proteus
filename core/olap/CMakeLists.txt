project(olap VERSION 0.1 LANGUAGES CXX)

# Pure, regular C++ files
set(olap_cxx
        common/common.cpp
        expressions/binary-operators.cpp
        expressions/expressions-dot-evaluator.cpp
        expressions/expressions-flusher.cpp
        expressions/expressions-generator.cpp
        expressions/expressions-hasher.cpp
        expressions/expressions.cpp
        expressions/path.cpp
        operators/cpu-to-gpu.cpp
        operators/dict-scan.cpp
        operators/flush.cpp
        operators/gpu/gmonoids.cpp
        operators/gpu/gpu-hash-group-by-chained.cpp
        operators/gpu/gpu-hash-join-chained.cpp
        operators/gpu/gpu-reduce.cpp
        operators/gpu/gpu-sort.cpp
        operators/gpu/gpu-to-cpu.cpp
        operators/hash-group-by-chained.cpp
        operators/hash-join-chained.cpp
        operators/hash-rearrange-buffered.cpp
        operators/hash-rearrange.cpp
        operators/join.cpp
        operators/materializer-expr.cpp
        operators/nest-opt.cpp
        operators/nest.cpp
        operators/null-filter.cpp
        operators/outer-unnest.cpp
        operators/packet-zip.cpp
        operators/print.cpp
        operators/project.cpp
        operators/radix-join.cpp
        operators/radix-nest.cpp
        operators/reduce-opt.cpp
        operators/relbuilder.cpp
        operators/root.cpp
        operators/router.cpp
        operators/scan.cpp
        operators/select.cpp
        operators/sort.cpp
        operators/split.cpp
        operators/unionall.cpp
        operators/unnest.cpp
        operators/mem-broadcast-device.cpp
        operators/mem-move-device.cpp
        operators/mem-move-local-to.cpp
        operators/block-to-tuples.cpp
        operators/gpu/gpu-hash-rearrange.cpp
        operators/gpu/gpu-partitioned-hash-join-chained.cpp
        plan/plan-parser.cpp
        plan/prepared-statement.cpp
        plan/query-result.cpp
        plugins/binary-block-plugin.cpp
        plugins/binary-col-plugin.cpp
        plugins/binary-internal-plugin.cpp
        plugins/binary-row-plugin.cpp
        plugins/csv-plugin-pm.cpp
        plugins/csv-plugin.cpp
        plugins/json-jsmn-plugin.cpp
        plugins/json-plugin.cpp
        plugins/output/plugins-output.cpp
        routing/affinitizers.cpp
        routing/routing-policy.cpp
        util/atois.cpp
        util/caching.cpp
        util/catalog.cpp
        util/context.cpp
        util/functions.cpp
        util/gpu/gpu-intrinsics.cpp
        util/jit/control-flow/if-statement.cpp
        util/jit/cpu-module.cpp
        util/jit/cpu-pipeline.cpp
        util/jit/gpu-module.cpp
        util/jit/gpu-pipeline.cpp
        util/jit/jit-module.cpp
        util/jit/pipeline.cpp
        util/parallel-context.cpp
        )

# C++ files requiring CUDA headers
set(olap_cxx_cu
        operators/gpu/gpu-partitioned-hash-join-chained-helpers.cpp
        )


foreach(target ${olap_cu})
    set_source_files_properties(${target} PROPERTIES LANGUAGE CXX)
endforeach()

foreach(target ${olap_cxx_cu} ${olap_cu})
    set_source_files_properties(${target} PROPERTIES COMPILE_FLAGS ${CUDA_CXXFLAGS})
endforeach()

#-----------------------------------------------------------------------------
# LSC: Hopefully we will soon be able to remove this section, where we call
#      straight to NVCC
set(CUDA_WORKDIR ${CMAKE_CURRENT_BINARY_DIR}/cubin)
file(MAKE_DIRECTORY ${CUDA_WORKDIR})

add_library(olap SHARED
        ${olap_cxx_cu}
        ${olap_cxx}
        )


#set_target_properties(olap PROPERTIES
#        LINK_FLAGS "-Wl,--no-allow-shlib-undefined -Wl,-z,defs ")

# Define headers for this library. PUBLIC headers are used for
# compiling the library, and will be added to consumers' build
# paths.
target_include_directories(olap
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR})

# If we have compiler requirements for this library, list them
# here
target_compile_features(olap
        PUBLIC cxx_std_17)

# Depend on a library that we defined in the top-level file
target_link_libraries(olap
    PUBLIC
        ${llvm_libs}
        codegen::codegen
        cli-flags)

# 'make install' to the correct locations (provided by GNUInstallDirs).
install(TARGETS olap EXPORT ${PROJECT_NAME}Targets
        ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})  # This is for Windows

# This makes the project importable from the install directory
# Put config file in per-project dir (name MUST match), can also
# just go into 'cmake'.
install(EXPORT ${PROJECT_NAME}Targets
        FILE ${PROJECT_NAME}Targets.cmake
        NAMESPACE ${PROJECT_NAME}::
        DESTINATION lib/cmake/${PROJECT_NAME})

# This makes the project importable from the build directory
export(TARGETS olap FILE ${PROJECT_NAME}Config.cmake)

add_library(codegen::olap ALIAS olap)
#enable_testing()
#add_test(UT codegen-tests) # Add codegen-specific tests?
