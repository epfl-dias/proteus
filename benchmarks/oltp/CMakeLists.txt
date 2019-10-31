# Pure, regular C++ files
set(txn_bench_cxx
	bench-cli-flags.cpp
    #tpcc.cpp
    tpcc_64.cpp
#    micro_ssb.cpp
    )


include_directories(
	"${PROJECT_SOURCE_DIR}"
	SYSTEM "${PROJECT_SOURCE_DIR}/../../engines/oltp/engine"
	SYSTEM "${PROJECT_SOURCE_DIR}/../../engines/oltp/lib"
	SYSTEM "${PROJECT_SOURCE_DIR}/../../codegen"
	"${CMAKE_INSTALL_PREFIX}/include"
	)

foreach(target
	bench-runner
	)
	add_executable(${target} ${target}.cpp ${txn_bench_cxx})
	target_link_libraries(${target} ${default_libs})
	install(TARGETS ${target}
		RUNTIME DESTINATION pelago
		LIBRARY DESTINATION lib
		ARCHIVE DESTINATION lib
		)
endforeach(target)


# project(oltp-bench VERSION 0.1)




# #-----------------------------------------------------------------------
# # Which binaries to build:
# #-----------------------------------------------------------------------
# set(OLTP_BENCH_MAINS bench-runner)



# string(STRIP "${WARNING_FLAGS} -Wno-atomic-implicit-seq-cst" WARNING_FLAGS)
# string(STRIP "${WARNING_FLAGS} -Wno-extra-semi-stmt" WARNING_FLAGS)
# string(STRIP "${WARNING_FLAGS} -Wno-unused-macros" WARNING_FLAGS)



# #-----------------------------------------------------------------------
# # Setup the library list, as the order is important do it last
# set(default_libs ${default_libs} codegen )
# set(default_libs ${default_libs} aeolus_engine )
# set(default_libs ${default_libs} -lnuma )
# set(default_libs ${default_libs} ${LLVM_LIBS} -lLLVMSupport)
# set(default_libs ${default_libs} ${CUDA_LIBS})
# set(default_libs ${default_libs} gflags::gflags ${GLOG} gflags::gflags)
# set(default_libs ${default_libs} -lc++abi -lpthread -lrt)



# #-----------------------------------------------------------------------
# # Intel VTune profile support
# find_library(VTUNE ittnotify HINTS "/opt/intel/vtune_amplifier/lib64")
# if(VTUNE AND VTUNE_ENABLE)
# 	include_directories(
# 		SYSTEM "/opt/intel/vtune_amplifier/include"
# 		)
# 	set(default_libs ${default_libs} ${VTUNE} -ldl)
# endif()

# #-----------------------------------------------------------------------
# # Build our libraries
# #-----------------------------------------------------------------------

# include_directories(
# 	"${PROJECT_SOURCE_DIR}"
# 	SYSTEM "${PROJECT_SOURCE_DIR}/../../engines/oltp/engine"
# 	SYSTEM "${PROJECT_SOURCE_DIR}/../../engines/oltp/lib"
# 	SYSTEM "${PROJECT_SOURCE_DIR}/../../codegen"

# 	# Let's see if we pull in headers which requires this, if not let's drop the whole block
# 	# FIXME: system libraries should not be included with quotes...
# 	# Add local (for use with ""-style includes) path to the sysroot
# 	"${CMAKE_INSTALL_PREFIX}/include"
# 	)

# # Add the warning flags
# string(STRIP "${CMAKE_CXX_FLAGS} ${WARNING_FLAGS}" CMAKE_CXX_FLAGS)
# string(STRIP "${CMAKE_C_FLAGS} ${WARNING_FLAGS}" CMAKE_C_FLAGS)

# #-----------------------------------------------------------------------
# # Build our executables
# #-----------------------------------------------------------------------


# foreach(target ${OLTP_BENCH_MAINS})
# 	add_executable(aeolus-${target} ${txn_bench_cxx}.cpp -fPIC)
# 	target_link_libraries(aeolus-${target} ${default_libs})
# 	install(TARGETS aeolus-${target}
# 		RUNTIME DESTINATION aeolus
# 		LIBRARY DESTINATION lib
# 		ARCHIVE DESTINATION lib
# 		)
# endforeach(target)

# #-----------------------------------------------------------------------

# # Unit-tests
# # if(UNIT_TESTS)
# # 	add_definitions(-DTEST_INPUTS="inputs")
# # 	add_definitions(-DTEST_OUTPUTS="outputs")
# # 	add_subdirectory(tests)
# # endif()


