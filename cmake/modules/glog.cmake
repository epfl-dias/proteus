set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(WITH_THREADS ON CACHE BOOL "" FORCE)
set(WITH_GFLAGS ON CACHE BOOL "" FORCE)

# Fix glog's find_package(GFLAG) by disabling find_package for gflags and
# setting up the directory paths
get_target_property(gflags_INCLUDE_DIR gflags::gflags INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(gflags_LIBRARIES gflags::gflags LIBRARIES)
macro(find_package)
	set(as_subproject gflags)
	if (NOT "${ARGV0}" IN_LIST as_subproject)
		_find_package(${ARGV})
	endif()
endmacro()

include(external/CMakeLists.txt.glog.in)

# Make glog appear as system library to avoid header warnings
get_target_property(glog_INCLUDE_DIR glog INTERFACE_INCLUDE_DIRECTORIES)
target_include_directories(glog SYSTEM PUBLIC ${glog_INCLUDE_DIR})
add_library(glog::glog ALIAS glog)

