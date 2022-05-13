find_package(CUDA)

if(CUDA_FOUND)
	# TODO: the CUDA package has been deprecated and replaced by native support for the language.
	# 		We should consider replacing it with the correct usage of enable_language(CUDA),
	# 		CMAKE_CUDA_COMPILER, CMAKE_CUDA_FLAGS etc
	if(DEFINED _NVCC_ARCHS)
		list(REMOVE_DUPLICATES _NVCC_ARCHS)

		CUDA_SELECT_NVCC_ARCH_FLAGS(CUDA_CU_ARCHS ${_NVCC_ARCHS})
	else(DEFINED _NVCC_ARCHS)
		CUDA_SELECT_NVCC_ARCH_FLAGS(CUDA_CU_ARCHS Pascal Volta Ampere 8.6)
	endif(DEFINED _NVCC_ARCHS)

	# Create arguments for nvcc and clang
	string(REGEX REPLACE " " [[;]] CUDA_CU_ARCHS_readable "${CUDA_CU_ARCHS_readable}")
	foreach(line ${CUDA_CU_ARCHS_readable})
		# Clang always includes PTX in the generated library for forward compatibility.
		# On the other hand, in nvcc it's optional.
		# So, if CUDA_CU_ARCHS contains a directive to include the PTX we should ignore it,
		# otherwise, clang complains.
		# DO NOT remove them from nvcc's flags!

		# Architecture names that start with "compute_" are nvcc directives to generate PTX
		if (NOT ${line} MATCHES "compute_*")
			list(APPEND CUDA_CXX_ARCHS "--cuda-gpu-arch=${line}")
		endif()
	endforeach(line)

	# C++ & CUDA hybrid files
	string(STRIP "${CUDA_CXXFLAGS} -Wno-format-pedantic -stdlib=libc++" CUDA_CXXFLAGS)
	string(STRIP "${CUDA_CXXFLAGS} -x cuda" CUDA_CXXFLAGS)
	string(STRIP "${CUDA_CXXFLAGS} --cuda-path=${CUDA_TOOLKIT_ROOT_DIR}" CUDA_CXXFLAGS)
	# Enable llvm::*->dump()
#	string(STRIP "${CUDA_CXXFLAGS} -DLLVM_ENABLE_DUMP" CUDA_CXXFLAGS)

	# Convert the list to a string
	string(REGEX REPLACE [[;]] " " CUDA_CXX_ARCHS "${CUDA_CXX_ARCHS}")
	string(REGEX REPLACE [[;]] " " PROTEUS_FORCE_CPPSTD_FOR_CUDA "${PROTEUS_FORCE_CPPSTD_FOR_CUDA}")
	string(STRIP "${CUDA_CXXFLAGS} ${CUDA_CXX_ARCHS}" CUDA_CXXFLAGS)
	string(STRIP "${CUDA_CXXFLAGS} -DMAXRREGCOUNT=32" CUDA_CXXFLAGS)
	string(STRIP "${CUDA_CXXFLAGS} ${PROTEUS_FORCE_CPPSTD_FOR_CUDA}" CUDA_CXXFLAGS)

	# Because of how the "command" function works, we have to use a
	# CMAKE list, and not a string for CUDA_CUFLAGS

	set(CUDA_LIBS -lnvToolsExt -lcuda ${CUDA_LIBRARIES}
			-lnvidia-ml
			-lcupti
			${CUDA_cudadevrt_LIBRARY} -lnvvm)

	include_directories(
		SYSTEM ${CUDA_INCLUDE_DIRS}
		)

	link_directories(
		"${CUDA_TOOLKIT_ROOT_DIR}/lib64"
		"${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs"
		"${CUDA_TOOLKIT_ROOT_DIR}/nvvm/lib64"
		)

	set(CUDA "CUDA-FOUND")
else()
	message(FATAL_ERROR "Warning: Building without support for GPUs (nvcc not
	detected)")

	set(CUDA_CXXFLAGS -x c++ -DNCUDA)
	set(CUDA_LIBS "")

	set(CUDA "CUDA-NOTFOUND")
endif()


