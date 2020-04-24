project(ssb100_bloom VERSION 0.1 LANGUAGES CXX)

# Pure, regular C++ files
set(${PROJECT_NAME}_cxx
        q1_1.sql.json.cpp
        q1_2.sql.json.cpp
        q1_3.sql.json.cpp
        q2_1.sql.json.cpp
        q2_2.sql.json.cpp
        q2_3.sql.json.cpp
        q3_1.sql.json.cpp
        q3_2.sql.json.cpp
        q3_3.sql.json.cpp
        q3_4.sql.json.cpp
        q4_1.sql.json.cpp
        q4_2.sql.json.cpp
        q4_3.sql.json.cpp
        )

add_library(${PROJECT_NAME} SHARED
        ${${PROJECT_NAME}_cxx}
        )

#set_target_properties(olap PROPERTIES
#        LINK_FLAGS "-Wl,--no-allow-shlib-undefined -Wl,-z,defs ")

# Define headers for this library. PUBLIC headers are used for
# compiling the library, and will be added to consumers' build
# paths.
target_include_directories(${PROJECT_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )



# Depend on a library that we defined in the top-level file
target_link_libraries(${PROJECT_NAME}
    PUBLIC
        codegen::olap)

# 'make install' to the correct locations (provided by GNUInstallDirs).
install(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}Targets
        ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})  # This is for Windows

# This makes the project importable from the install directory
# Put config file in per-project dir (name MUST match), can also
# just go into 'cmake'.
install(EXPORT ${PROJECT_NAME}Targets
        FILE ${PROJECT_NAME}Targets.cmake
        NAMESPACE ${PROJECT_NAME}::
        DESTINATION lib/cmake/${PROJECT_NAME})

# This makes the project importable from the build directory
export(TARGETS ${PROJECT_NAME} FILE ${PROJECT_NAME}Config.cmake)

add_library(queries::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
