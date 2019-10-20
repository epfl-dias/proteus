set(codegen_cxx
    queries.cpp
    query_sequence.cpp
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
