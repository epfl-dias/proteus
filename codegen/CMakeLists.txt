# Pure, regular C++ files
set(codegen_cxx
    common/common.cpp
    common/llvm-dump.cpp
    common/gpu/gpu-common.cpp
    common/symantec-config.cpp
    expressions/binary-operators.cpp
    expressions/expressions-dot-evaluator.cpp
    expressions/expressions-flusher.cpp
    expressions/expressions-generator.cpp
    expressions/expressions-hasher.cpp
    expressions/expressions.cpp
    expressions/path.cpp
    memory/memory-allocator.cpp
    memory/memory-manager.cpp
    operators/cpu-to-gpu.cpp
    operators/dict-scan.cpp
    operators/exchange.cpp
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
    operators/reduce-nopred.cpp
    operators/reduce-opt.cpp
    operators/reduce.cpp
    operators/root.cpp
    operators/scan.cpp
    operators/select.cpp
    operators/sort.cpp
    operators/split.cpp
    operators/unionall.cpp
    operators/unnest.cpp
    plan/plan-parser.cpp
    plugins/binary-col-plugin.cpp
    plugins/binary-internal-plugin.cpp
    plugins/binary-row-plugin.cpp
    plugins/csv-plugin-pm.cpp
    plugins/csv-plugin.cpp
    plugins/json-jsmn-plugin.cpp
    plugins/json-plugin.cpp
    plugins/output/plugins-output.cpp
    plugins/scan-to-blocks-sm-plugin.cpp
    storage/storage-manager.cpp
    topology/affinity_manager.cpp
    topology/topology.cpp
    util/atois.cpp
    util/caching.cpp
    util/catalog.cpp
    util/context.cpp
    util/functions.cpp
    util/gpu/gpu-intrinsics.cpp
    util/jit/cpu-module.cpp
    util/jit/cpu-pipeline.cpp
    util/jit/gpu-module.cpp
    util/jit/gpu-pipeline.cpp
    util/jit/jit-module.cpp
    util/jit/pipeline.cpp
    util/parallel-context.cpp
    util/radix/aggregations/radix-aggr.cpp
    util/radix/joins/radix-join.cpp
    values/expressionTypes.cpp
    )

# C++ files requiring CUDA headers
set(codegen_cxx_cu
    operators/block-to-tuples.cpp
    operators/gpu/gpu-hash-rearrange.cpp
    operators/gpu/gpu-partitioned-hash-join-chained.cpp
    operators/mem-broadcast-device.cpp
    operators/mem-move-device.cpp
    operators/mem-move-local-to.cpp
    )

# Pure CUDA files
set(codegen_cu
# LSC: 30.11.2018
#       Work around issue in:
#         /usr/local/cuda/targets/x86_64-linux/include/thrust/system/cuda/detail/cub/util_debug.cuh 117-136
#       Which fails with va_printf undefined.
    #util/sort/gpu/sort-util.cu

# LSC: 30.11.2018
#       For now we build two times this, as otherwise we get linking and/or
#       runtime error as LLVM cannot yet generate relocatable ptx code.
        memory/buffer-manager.cu
    )

foreach(target ${codegen_cu})
    set_source_files_properties(${target} PROPERTIES LANGUAGE CXX)
endforeach()

foreach(target ${codegen_cxx_cu} ${codegen_cu})
    set_source_files_properties(${target} PROPERTIES COMPILE_FLAGS ${CUDA_CXXFLAGS})
endforeach()

#-----------------------------------------------------------------------------
# LSC: Hopefully we will soon be able to remove this section, where we call
#      straight to NVCC
set(CUDA_WORKDIR ${CMAKE_CURRENT_BINARY_DIR}/cubin)
file(MAKE_DIRECTORY ${CUDA_WORKDIR})

if(CUDA_FOUND)
    set(codegen_cu_objs
        "${CUDA_WORKDIR}/buffer-manager.fatbin.o"
        "${CUDA_WORKDIR}/sort-util.o"
    )
endif()

add_custom_command(OUTPUT ${CUDA_WORKDIR}/sort-util.o
    COMMAND ${CUDA_NVCC_EXECUTABLE} ${CUDA_CUFLAGS}
        -c ${CMAKE_CURRENT_SOURCE_DIR}/util/sort/gpu/sort-util.cu
        -o sort-util.o
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/util/sort/gpu/sort-util.cu
    IMPLICIT_DEPENDS CXX ${CMAKE_CURRENT_SOURCE_DIR}/util/sort/gpu/sort-util.cu
    WORKING_DIRECTORY ${CUDA_WORKDIR}
    COMMENT "Building CUDA object ${CUDA_WORKDIR}/sort-util.o"
)

add_custom_command(OUTPUT ${CUDA_WORKDIR}/buffer-manager.fatbin
    COMMAND ${CUDA_NVCC_EXECUTABLE} ${CUDA_CUFLAGS}
        --keep --keep-dir ${CUDA_WORKDIR} -rdc true -fatbin
        ${CMAKE_CURRENT_SOURCE_DIR}/memory/buffer-manager.cu
        -o ${CUDA_WORKDIR}/buffer-manager.fatbin
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/memory/buffer-manager.cu
    IMPLICIT_DEPENDS CXX ${CMAKE_CURRENT_SOURCE_DIR}/memory/buffer-manager.cu
    WORKING_DIRECTORY ${CUDA_WORKDIR}
    COMMENT "Building CUDA object ${CUDA_WORKDIR}/buffer-manager.o"
)

add_custom_command(OUTPUT ${CUDA_WORKDIR}/buffer-manager.fatbin.o
    COMMAND ld -r -b binary
        ${CUDA_WORKDIR}/buffer-manager*.cubin
        -o ${CUDA_WORKDIR}/buffer-manager.fatbin.o
    DEPENDS ${CUDA_WORKDIR}/buffer-manager.fatbin
    WORKING_DIRECTORY ${CUDA_WORKDIR}
    COMMENT "Linking CUDA object ${CUDA_WORKDIR}/buffer-manager.cubin.o"
)
#-----------------------------------------------------------------------------

add_library(codegen SHARED
        ${codegen_cxx_cu}
        ${codegen_cu}
        ${codegen_cu_objs}
        ${codegen_cxx}
    )
target_link_libraries(codegen ${LLVM_LIBS} jsmn -lnuma)
install(TARGETS codegen
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    )

install(DIRECTORY DESTINATION pelago/generated_code)
#enable_testing()
#add_test(UT codegen-tests) # Add codegen-specific tests?

