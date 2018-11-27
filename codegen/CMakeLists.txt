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
	topology/*.cpp
	multigpu/common.cpp
	multigpu/buffer_manager.cu
	)
foreach(target
	${codegen_src}
	)
	SET_SOURCE_FILES_PROPERTIES(${target} PROPERTIES LANGUAGE CXX)
endforeach(target)

get_property(inc_dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
string(REGEX REPLACE "([^;]+)" "-I\\1" inc_dirs2 "${inc_dirs}")

if (NOT (${CUDA} STREQUAL "CUDA-NOTFOUND"))
add_custom_command(OUTPUT sort-util.o
                   COMMAND nvcc ${CUDA_CUFLAGS} -O3 --generate-line-info ${inc_dirs2} -c ${CMAKE_CURRENT_SOURCE_DIR}/util/sort/gpu/sort-util.cu -o ${CMAKE_CURRENT_BINARY_DIR}/sort-util.o
                   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/util/sort/gpu/sort-util.cu ${CMAKE_CURRENT_SOURCE_DIR}/common/gpu/gpu-common.hpp ${CMAKE_CURRENT_SOURCE_DIR}/../common/common.hpp ${CMAKE_CURRENT_SOURCE_DIR}/util/sort/gpu/sort-util.cuh
                   # WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                   )

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

add_custom_target(buffer_manager     DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/buffer_manager.cubin)
add_custom_target(buffer_manager_obj DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/buffer_manager.cubin.o)


add_library(codegen SHARED ${codegen_src} ${CMAKE_CURRENT_BINARY_DIR}/buffer_manager.cubin.o ${CMAKE_CURRENT_BINARY_DIR}/sort-util.o)
else()
    add_library(codegen SHARED ${codegen_src})
endif()

set_target_properties(codegen PROPERTIES COMPILE_FLAGS ${LLVM_CXXFLAGS})

target_link_libraries(codegen jsmn)
if (NOT (${CUDA} STREQUAL "CUDA-NOTFOUND"))
    add_dependencies(codegen buffer_manager_obj)
endif()

install(TARGETS codegen
	RUNTIME DESTINATION bin
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib
	)

install(DIRECTORY DESTINATION raw/generated_code)
