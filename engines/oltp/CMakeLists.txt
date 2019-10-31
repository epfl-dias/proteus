project(aeolus VERSION 0.1)

#-----------------------------------------------------------------------
# Which binaries to build:
#-----------------------------------------------------------------------
# Simply comment the lines of this section to skip some of the binaries
# to build.
set(OLTP_MAINS server)

#-----------------------------------------------------------------------
# Build our libraries
#-----------------------------------------------------------------------

include_directories(
	"${CMAKE_CURRENT_SOURCE_DIR}"
	"${CMAKE_CURRENT_SOURCE_DIR}/engine"
	SYSTEM "${CMAKE_CURRENT_SOURCE_DIR}/lib"
	)

#-----------------------------------------------------------------------
# Build our executables
#-----------------------------------------------------------------------
add_subdirectory(engine)

set(default_libs ${default_libs} aeolus_engine )
foreach(target ${OLTP_MAINS})
	add_executable(aeolus-${target} ${target}.cpp)
	target_link_libraries(aeolus-${target} ${default_libs})
	install(TARGETS aeolus-${target}
		RUNTIME DESTINATION aeolus
		LIBRARY DESTINATION lib
		ARCHIVE DESTINATION lib
		)
endforeach(target)

add_subdirectory(adaptors)

#-----------------------------------------------------------------------

# Unit-tests
if(UNIT_TESTS)
	add_subdirectory(tests)
endif()
