project(deprecated-olap-apps VERSION 0.1 LANGUAGES CXX)

set(DMAINS
        main
        main-caches
        main-nest
        main-str
        main-reduce
        )

foreach(target ${DMAINS})
    add_executable(deprecated-proteus${target} ${target}.cpp)
    target_link_libraries(deprecated-proteus${target} proteus::olap)
    install(TARGETS deprecated-proteus${target}
            ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
            LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR}
            )
endforeach(target)
