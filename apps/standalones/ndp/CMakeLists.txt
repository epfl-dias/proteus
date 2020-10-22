get_filename_component(STANDALONE_DIR ${CMAKE_CURRENT_SOURCE_DIR} NAME)

project(${STANDALONE_DIR} VERSION 0.1 LANGUAGES CXX)

set(ndp_cxx
        ndp-cli-flags.cpp
        )

set(MAINS
        ${STANDALONE_DIR}
        )

foreach(target ${MAINS})
    add_executable(proteus${target} ${target}.cpp ${ndp_cxx})
    target_link_libraries(proteus${target} codegen::olap_flags codegen::olap proteus::runtime)
    install(TARGETS proteus${target}
            ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
            LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR}
            )
endforeach(target)

