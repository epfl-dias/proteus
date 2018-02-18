file(GLOB codegen_src
	common/*.cpp
	common/gpu/*.cpp
	util/*.cpp
	util/jit/*.cpp
	util/gpu/*.cpp
	util/radix/*.cpp
	util/radix/joins/*.cpp
	util/radix/aggregations/*.cpp
	values/*.cpp
	expressions/*.cpp
	plugins/*.cpp
	plugins/output/*.cpp
	operators/*.cpp
	operators/gpu/*.cpp
	memory/*.cpp
	plan/*.cpp
	postgres/*.cpp
	storage/*.cpp
	# multigpu/operators/*.cu
	# multigpu/buffer_pool.cu
	# # src/output_composer.cu
	multigpu/common.cpp
    multigpu/buffer_manager.cu
	# multigpu/common.cu
	)
foreach(target
	${codegen_src}
	)

# if (${CUDA} STREQUAL "CUDA-NOTFOUND")
# list(REMOVE_ITEM codegen_src ${CMAKE_CURRENT_SOURCE_DIR}/multigpu/operators/gpu_to_cpu.cu)
# endif()

	SET_SOURCE_FILES_PROPERTIES(${target} PROPERTIES LANGUAGE CXX)
endforeach(target)

# # SET_SOURCE_FILES_PROPERTIES(device_funcs_link.o EXTERNAL_OBJECT TRUE GENERATED true)

# # list(REMOVE_ITEM multigpu_src ${CMAKE_CURRENT_SOURCE_DIR}/src/output_composer.cu)
# list(REMOVE_ITEM codegen_src ${CMAKE_CURRENT_SOURCE_DIR}/multigpu/operators/select.cu)
# list(REMOVE_ITEM codegen_src ${CMAKE_CURRENT_SOURCE_DIR}/multigpu/operators/select2.cu)
# list(REMOVE_ITEM codegen_src ${CMAKE_CURRENT_SOURCE_DIR}/multigpu/operators/materializer.cu)
# list(REMOVE_ITEM codegen_src ${CMAKE_CURRENT_SOURCE_DIR}/multigpu/operators/operator.cu)
# list(REMOVE_ITEM codegen_src ${CMAKE_CURRENT_SOURCE_DIR}/multigpu/operators/aggregation.cu)
# list(REMOVE_ITEM codegen_src ${CMAKE_CURRENT_SOURCE_DIR}/multigpu/operators/select3.cu)
# list(REMOVE_ITEM codegen_src ${CMAKE_CURRENT_SOURCE_DIR}/multigpu/operators/select_v.cu)
# list(REMOVE_ITEM codegen_src ${CMAKE_CURRENT_SOURCE_DIR}/multigpu/operators/union_all_cpu.cu)

# # FIXME: multigpu::gpu_to_cpu is using thrust::tuple and clang produces errors when compiling thrust
# list(REMOVE_ITEM codegen_src ${CMAKE_CURRENT_SOURCE_DIR}/multigpu/operators/gpu_to_cpu.cu)

get_property(inc_dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
string(REGEX REPLACE "([^;]+)" "-I\\1" inc_dirs2 "${inc_dirs}")

# add_custom_command(OUTPUT device_funcs.cubin
#                    COMMAND nvcc ${CUDA_CXXFLAGS} -O3 --generate-line-info ${inc_dirs2} -dc --cubin ${CMAKE_CURRENT_SOURCE_DIR}/device_funcs.cu -o ${CMAKE_CURRENT_BINARY_DIR}/device_funcs.cubin
#                    DEPENDS device_funcs.cu
#                    # WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
#                    )

# add_custom_command(OUTPUT device_funcs.cubin.o
#                    COMMAND ld -r -b binary device_funcs.cubin -o ${CMAKE_CURRENT_BINARY_DIR}/device_funcs.cubin.o
#                    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/device_funcs.cubin
#                    # WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
#                    )

# add_custom_target(device_funcs     DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/device_funcs.cubin)
# add_custom_target(device_funcs_obj DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/device_funcs.cubin.o)

if (NOT (${CUDA} STREQUAL "CUDA-NOTFOUND"))
add_custom_command(OUTPUT buffer_manager.cubin
                   COMMAND nvcc ${CUDA_CUFLAGS} -O3 --generate-line-info ${inc_dirs2} -dc --cubin ${CMAKE_CURRENT_SOURCE_DIR}/multigpu/buffer_manager.cu -o ${CMAKE_CURRENT_BINARY_DIR}/buffer_manager.cubin
                   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/multigpu/buffer_manager.cu ${CMAKE_CURRENT_SOURCE_DIR}/multigpu/buffer_manager.cuh ${CMAKE_CURRENT_SOURCE_DIR}/multigpu/threadsafe_device_stack.cuh ${CMAKE_CURRENT_SOURCE_DIR}/multigpu/threadsafe_stack.cuh
                   # WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                   )

add_custom_command(OUTPUT buffer_manager.cubin.o
                   COMMAND ld -r -b binary buffer_manager.cubin -o ${CMAKE_CURRENT_BINARY_DIR}/buffer_manager.cubin.o
                   DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/buffer_manager.cubin
                   # WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                   )

# add_subdirectory(multigpu)
endif()

add_custom_target(buffer_manager     DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/buffer_manager.cubin)
add_custom_target(buffer_manager_obj DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/buffer_manager.cubin.o)

if (${CUDA} STREQUAL "CUDA-NOTFOUND")
    add_library(codegen SHARED ${codegen_src})
else()
    add_library(codegen SHARED ${codegen_src} ${CMAKE_CURRENT_BINARY_DIR}/buffer_manager.cubin.o)
endif()
set_target_properties(codegen PROPERTIES COMPILE_FLAGS ${LLVM_CXXFLAGS})

target_link_libraries(codegen jsmn) # multigpu)
if (NOT (${CUDA} STREQUAL "CUDA-NOTFOUND"))
    add_dependencies(codegen buffer_manager_obj)
endif()

# add_dependencies(codegen device_funcs_obj)

install(TARGETS codegen
	RUNTIME DESTINATION bin
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib
	)

install(DIRECTORY DESTINATION raw/generated_code)

# install(FILES ${CMAKE_CURRENT_BINARY_DIR}/device_funcs.cubin
# 	DESTINATION res
# 	# LIBRARY DESTINATION lib
# 	# ARCHIVE DESTINATION lib
# 	)
