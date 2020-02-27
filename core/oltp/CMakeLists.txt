project(aeolus VERSION 0.1)

#-----------------------------------------------------------------------
# Which binaries to build:
#-----------------------------------------------------------------------
# Simply comment the lines of this section to skip some of the binaries
# to build.
set(OLTP_MAINS)

set(OLTP_CLI_FLAGS
        )


foreach (target ${OLTP_CLI_FLAGS})
    set_source_files_properties(${target} PROPERTIES COMPILE_FLAGS ${CUDA_CXXFLAGS})
endforeach ()


#-----------------------------------------------------------------------
# Build our libraries
#-----------------------------------------------------------------------
#
#include_directories(
#	"${CMAKE_CURRENT_SOURCE_DIR}"
#	"${CMAKE_CURRENT_SOURCE_DIR}/engine"
#	SYSTEM "${CMAKE_CURRENT_SOURCE_DIR}/lib"
#	)

#-----------------------------------------------------------------------
# Build our executables
#-----------------------------------------------------------------------
add_subdirectory(lib/libcuckoo)
add_subdirectory(engine)

foreach (target ${OLTP_MAINS})
    set_source_files_properties(${target}.cpp PROPERTIES COMPILE_FLAGS ${CUDA_CXXFLAGS})

    add_executable(aeolus-${target} ${target}.cpp ${OLTP_CLI_FLAGS})
    target_link_libraries(aeolus-${target} PUBLIC aeolus::engine)
    install(TARGETS aeolus-${target}
            RUNTIME DESTINATION pelago
            LIBRARY DESTINATION lib
            ARCHIVE DESTINATION lib
            )
endforeach (target)

add_subdirectory(adaptors)

#-----------------------------------------------------------------------

# Unit-tests
if (UNIT_TESTS)
    add_subdirectory(tests)
endif ()
