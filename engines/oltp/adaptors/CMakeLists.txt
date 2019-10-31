# Pure, regular C++ files
set(adaptors_cxx
    aeolus-plugin.cpp
    )



#-----------------------------------------------------------------------------

add_library(aeolus_plugin SHARED
        ${adaptors_cxx}
    )
target_link_libraries(aeolus_plugin aeolus_engine codegen)
install(TARGETS aeolus_plugin
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    )


#enable_testing()
#add_test(UT codegen-tests) # Add codegen-specific tests?


