project(aeolus VERSION 0.1)

#-----------------------------------------------------------------------
# Which binaries to build:
#-----------------------------------------------------------------------
# Simply comment the lines of this section to skip some of the binaries
# to build.
set(OLTP_MAINS server)



string(STRIP "${WARNING_FLAGS} -Wno-atomic-implicit-seq-cst" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-extra-semi-stmt" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-unused-macros" WARNING_FLAGS)



#-----------------------------------------------------------------------
# Setup the library list, as the order is important do it last
set(default_libs ${default_libs} codegen )
set(default_libs ${default_libs} -lnuma )
set(default_libs ${default_libs} ${LLVM_LIBS} -lLLVMSupport)
set(default_libs ${default_libs} ${CUDA_LIBS})
set(default_libs ${default_libs} gflags::gflags ${GLOG} gflags::gflags)
set(default_libs ${default_libs} -lc++abi -lpthread -lrt)



#-----------------------------------------------------------------------
# Intel VTune profile support
find_library(VTUNE ittnotify HINTS "/opt/intel/vtune_amplifier/lib64")
if(VTUNE AND VTUNE_ENABLE)
	include_directories(
		SYSTEM "/opt/intel/vtune_amplifier/include"
		)
	set(default_libs ${default_libs} ${VTUNE} -ldl)
endif()

#-----------------------------------------------------------------------
# Build our libraries
#-----------------------------------------------------------------------

include_directories(
	"${PROJECT_SOURCE_DIR}"
	"${PROJECT_SOURCE_DIR}/engine"
	SYSTEM "${PROJECT_SOURCE_DIR}/lib"
	SYSTEM "${PROJECT_SOURCE_DIR}/../../codegen"

	# Let's see if we pull in headers which requires this, if not let's drop the whole block
	# FIXME: system libraries should not be included with quotes...
	# Add local (for use with ""-style includes) path to the sysroot
	"${CMAKE_INSTALL_PREFIX}/include"
	)

# Add the warning flags
string(STRIP "${CMAKE_CXX_FLAGS} ${WARNING_FLAGS}" CMAKE_CXX_FLAGS)
string(STRIP "${CMAKE_C_FLAGS} ${WARNING_FLAGS}" CMAKE_C_FLAGS)

#-----------------------------------------------------------------------
# Build our executables
#-----------------------------------------------------------------------
add_subdirectory(engine)

# set(default_libs ${default_libs} aeolus_engine )
# foreach(target ${OLTP_MAINS})
# 	add_executable(aeolus-${target} ${target}.cpp)
# 	target_link_libraries(aeolus-${target} ${default_libs})
# 	install(TARGETS aeolus-${target}
# 		RUNTIME DESTINATION aeolus
# 		LIBRARY DESTINATION lib
# 		ARCHIVE DESTINATION lib
# 		)
# endforeach(target)

if (HTAP)
add_subdirectory(adaptors)
endif(HTAP)

#-----------------------------------------------------------------------

# Unit-tests
if(UNIT_TESTS)
	add_definitions(-DTEST_INPUTS="inputs")
	add_definitions(-DTEST_OUTPUTS="outputs")
	add_subdirectory(tests)
endif()
