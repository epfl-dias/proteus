find_package(LLVM REQUIRED VERSION ${LLVM_REQUIRED_VERSION} CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# Check if LLVM is build as a monolithic shared library
if (${LLVM_LINK_LLVM_DYLIB}) # Usually in the CI and with pre-build	a LLVM
	message(STATUS "Using monolithic libLLVM.so")
	# If yes, then we should only build with libLLVM.so, to avoid double refers
	set(llvm_libs LLVM)
else() 						 # Usually in LLVM dev mode or Pelago
	message(STATUS "Using component-based LLVM")
	# If no, we can link with only the required components
	# LLVM APIs we use
#	set(LLVM_COMPONENTS aggressiveinstcombine all all-targets analysis asmparser asmprinter binaryformat bitreader bitwriter codegen core coroutines coverage debuginfocodeview debuginfodwarf debuginfomsf debuginfopdb demangle dlltooldriver engine executionengine fuzzmutate globalisel instcombine instrumentation inteljitevents interpreter ipo irreader libdriver lineeditor linker lto mc mcdisassembler mcjit mcparser mirparser native nativecodegen nvptx nvptxasmprinter nvptxcodegen nvptxdesc nvptxinfo objcarcopts object objectyaml option orcjit passes profiledata runtimedyld scalaropts selectiondag support symbolize tablegen target transformutils vectorize windowsmanifest x86 x86asmparser x86asmprinter x86codegen x86desc x86disassembler x86info x86utils)
#	set(LLVM_COMPONENTS all)
#
#	llvm_map_components_to_libnames(llvm_libs ${LLVM_COMPONENTS})
	set(llvm_libs ${LLVM_AVAILABLE_LIBS})
endif()
message(STATUS "Linking with LLVM components: ${llvm_libs}")

# Check if LLVM depends on libc++ or libstdc++
set(USE_LIBCXX FALSE)

foreach(llvm_lib ${llvm_libs})
find_library(lib ${llvm_lib} PATHS "${LLVM_LIBRARY_DIR}" NO_DEFAULT_PATH REQUIRED)
get_prerequisites("${lib}" _prereqs FALSE TRUE "" "${LLVM_LIBRARY_DIR}")
if (_prereqs MATCHES "/libc\\+\\+")
	set(USE_LIBCXX TRUE)
	break()
endif()
endforeach()

if (USE_LIBCXX)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++")
	set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -stdlib=libc++")
	set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -stdlib=libc++")

	# Also add libc++ to the llvm_libs so that we get std::filesystem; apparently -stdlib=libc++ is not enough
	find_library(libc++ c++ PATHS "${LLVM_LIBRARY_DIR}" NO_DEFAULT_PATH REQUIRED)
	list(APPEND llvm_libs ${libc++})
	list(APPEND PROTEUS_FORCE_CPPSTD_FOR_CUDA "--std=c++20")
else()
	# Patch for GCC shipping libstdc++fs separately from libstdc++
	list(APPEND llvm_libs ${llvm_libs} stdc++fs)
	# Using c++20 with cuda and libstdc++-10-dev causes compilation issues
	list(APPEND PROTEUS_FORCE_CPPSTD_FOR_CUDA "--stdlib=libstdc++")
	list(APPEND PROTEUS_FORCE_CPPSTD_FOR_CUDA "--std=c++17")
	list(APPEND PROTEUS_FORCE_CPPSTD_FOR_CUDA "-Wno-error=zero-as-null-pointer-constant")
endif()

if (CMAKE_CXX_FLAGS MATCHES ".*-stdlib=libc\\+\\+.*")
	set(USE_LIBCXX ON)
else()
	set(USE_LIBCXX OFF)
endif()

message(STATUS "Using libc++: ${USE_LIBCXX}")

add_definitions(${LLVM_DEFINITIONS})
message(STATUS "Using LLVM definitions: ${LLVM_DEFINITIONS}")

link_directories(${LLVM_LIBRARY_DIR})
message(STATUS "Using LLVM library dir: ${LLVM_LIBRARY_DIR}")

add_library(LLVM_VIRTUAL_TARGET INTERFACE)

target_include_directories(LLVM_VIRTUAL_TARGET
    SYSTEM PUBLIC INTERFACE
		${LLVM_INCLUDE_DIRS}
)

target_link_libraries(LLVM_VIRTUAL_TARGET
	INTERFACE
		${llvm_libs}
)

# 'make install' to the correct locations (provided by GNUInstallDirs).
install(TARGETS LLVM_VIRTUAL_TARGET EXPORT LLVMVirtualConfig
		ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
			COMPONENT          LLVMVirtual_Development
		LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
			COMPONENT          LLVMVirtual_RunTime
			NAMELINK_COMPONENT LLVMVirtual_Development
		RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}  # This is for Windows
			COMPONENT          LLVMVirtual_RunTime
)

install(EXPORT LLVMVirtualConfig
	DESTINATION lib/cmake/${PROJECT_NAME}
		COMPONENT LLVMVirtual_Development
)
export(TARGETS LLVM_VIRTUAL_TARGET FILE LLVMVirtualConfig.cmake)

add_library(LLVM::LLVM ALIAS LLVM_VIRTUAL_TARGET)
