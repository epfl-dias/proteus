set(codegen_cxx
    ch/q1.cpp
    ch/q6.cpp
    micro/sum.cpp
    queries.cpp
    )

add_library(queries SHARED
        ${codegen_cxx}
    )
target_link_libraries(queries ${default_libs})
install(TARGETS queries
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    )
