project(olap-apps VERSION 0.1 LANGUAGES CXX)

set(MAINS
        main-server  # Interface to the executor
        )

foreach(target ${MAINS})
    add_executable(proteus${target} ${target}.cpp)
    target_link_libraries(proteus${target} codegen::olap_flags codegen::olap)
    install(TARGETS proteus${target}
            ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
            LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR}
            )
endforeach(target)

