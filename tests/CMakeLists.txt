foreach(target
	tests-joins
	tests-json
	tests-output
	tests-plan-parsing
	tests-multigpu-integration
	)
	add_executable(unit-${target} ${target}.cpp)
	set_target_properties(unit-${target} PROPERTIES COMPILE_FLAGS ${LLVM_CXXFLAGS})
	target_link_libraries(unit-${target} ${gtest_libs})
	install(TARGETS unit-${target}
		RUNTIME DESTINATION raw
		LIBRARY DESTINATION lib
		ARCHIVE DESTINATION lib
		)
endforeach(target)

# find_package(CUDA REQUIRED)
# set(CUDA_HOST_COMPILER "g++") #FIXME: workaround as nvcc does not support the current clang version

# find_program(CLANG_CXX_COMPILER "nvcc")
# set(CMAKE_CXX_COMPILER "nvcc")
# find_program(CLANG_C_COMPILER "nvcc")
# set(CMAKE_C_COMPILER "nvcc")

# foreach(target
# 	# tests-gpu2
# 	# tests-gpu-ssbm
# 	)
# 	# set(CMAKE_CXX_COMPILER "nvcc")
# 	# find_program(CLANG_C_COMPILER "nvcc")
# 	# set(CMAKE_C_COMPILER "nvcc")

# 	SET_SOURCE_FILES_PROPERTIES(${target}.cu PROPERTIES LANGUAGE CXX)
# 	add_executable(unit-${target} ${target}.cu)
# 	set_target_properties(unit-${target} PROPERTIES COMPILE_FLAGS "${LLVM_CXXFLAGS} -march=native")
# 	target_link_libraries(unit-${target} ${gtest_libs})
# 	install(TARGETS unit-${target}
# 		RUNTIME DESTINATION raw
# 		LIBRARY DESTINATION lib
# 		ARCHIVE DESTINATION lib
# 		)
# endforeach(target)

# find_program(CLANG_CXX_COMPILER "clang++")
# set(CMAKE_CXX_COMPILER "${CLANG_CXX_COMPILER}")
# find_program(CLANG_C_COMPILER "clang")
# set(CMAKE_C_COMPILER "${CLANG_C_COMPILER}")


# find_package(CUDA QUIET REQUIRED)

# set(CUDA_HOST_COMPILER g++)

# foreach(target
# 	tests-multigpu-integration
# 	)
# 	# SET_SOURCE_FILES_PROPERTIES(${target}.cu PROPERTIES LANGUAGE CXX)
# 	SET_SOURCE_FILES_PROPERTIES(${target}.cu PROPERTIES LANGUAGE CUDA)
# 	add_executable(unit-${target} ${target}.cu)
# 	target_compile_features(unit-${target} PUBLIC cxx_std_11)
# 	set_target_properties(unit-${target} PROPERTIES COMPILE_FLAGS ${LLVM_CXXFLAGS})
# 	set_target_properties(unit-${target} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# 	target_link_libraries(unit-${target} ${gtest_libs})
# 	install(TARGETS unit-${target}
# 		RUNTIME DESTINATION raw
# 		LIBRARY DESTINATION lib
# 		ARCHIVE DESTINATION lib
# 		)
# endforeach(target)

# One test to rule them all
file(GLOB gtest_src *.cpp *.cu)
add_executable(unit-tests ${gtest_src})
set_target_properties(unit-tests PROPERTIES COMPILE_FLAGS "${LLVM_CXXFLAGS} -march=native")
target_link_libraries(unit-tests ${gtest_libs})
install(TARGETS unit-tests
	RUNTIME DESTINATION raw
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib
	)

enable_testing()
add_test(UT unit-tests)
