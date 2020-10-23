get_filename_component(STANDALONE_DIR ${CMAKE_CURRENT_SOURCE_DIR} NAME)

project(${STANDALONE_DIR} VERSION 0.1 LANGUAGES CXX)

set(ndp_cxx
        ndp-cli-flags.cpp
        lib/ndp-common.cpp
        lib/cluster_command_provider.cpp
        lib/example-plans/alternative-paths.cpp
        lib/example-plans/scheduling-paths.cpp
        lib/example-plans/task-description.cpp
        lib/example-plans/task-offloading.cpp
        )

set(MAINS
        ${STANDALONE_DIR}
        ${STANDALONE_DIR}-commander
        )

foreach(target ${MAINS})
    add_executable(proteus${target} ${target}.cpp ${ndp_cxx})
    target_link_libraries(proteus${target} codegen::olap_flags proteus::runtime
        proteus::plan-parser proteus::command_provider)
    target_include_directories(proteus${target}
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
    )
    install(TARGETS proteus${target}
            ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
            LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR}
            )
endforeach(target)

