project(olap_prepared_queries VERSION 0.1 LANGUAGES CXX)

set(MAINS
        run-prepared-query  # Interface to the executor
        )

add_subdirectory(queries/ch100w)
add_subdirectory(queries/ssb100)
add_subdirectory(queries/ssb100_bloom)
add_subdirectory(queries/ssb1000)


foreach(target ${MAINS})
    add_executable(proteus${target} ${target}.cpp)
    target_link_libraries(proteus${target} proteus::platform proteus::olap
            codegen::olap_flags queries::ssb100 queries::ssb100_bloom queries::ssb1000)
    install(TARGETS proteus${target}
            ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
            LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR}
            )
endforeach(target)

