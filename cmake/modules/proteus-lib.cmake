set_target_properties(${PROJECT_NAME}
    PROPERTIES
        SOVERSION ${${PROJECT_NAME}_VERSION_MAJOR}
        VERSION   ${${PROJECT_NAME}_VERSION})

# Error on undefined symbols in shared libraries
set_target_properties(${PROJECT_NAME} PROPERTIES
    LINK_FLAGS "-Wl,--no-allow-shlib-undefined -Wl,-z,defs")

# Define headers for this library. PUBLIC headers are used for
# compiling the library, and will be added to consumers' build
# paths.
target_include_directories(${PROJECT_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )

include(GNUInstallDirs)
# 'make install' to the correct locations (provided by GNUInstallDirs).
install(TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}Targets
    RUNTIME   DESTINATION ${CMAKE_INSTALL_BINDIR}  # This is for Windows
              COMPONENT           ${PROTEUS_RUNTIME_COMPONENT}
    LIBRARY   DESTINATION ${CMAKE_INSTALL_LIBDIR}
              COMPONENT           ${PROTEUS_RUNTIME_COMPONENT}
              NAMELINK_COMPONENT  ${PROTEUS_DEVELOPMENT_COMPONENT}
    ARCHIVE   DESTINATION ${CMAKE_INSTALL_LIBDIR}
              COMPONENT           ${PROTEUS_DEVELOPMENT_COMPONENT}
# Do not use PUBLIC_HEADER as it can't deduce the public header structure
#    set_target_properties(olap PROPERTIES PUBLIC_HEADER ${olap_public_hxx})
#       PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    INCLUDES  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    COMPONENT   ${PROTEUS_DEVELOPMENT_COMPONENT})

# This makes the project importable from the install directory
# Put config file in per-project dir (name MUST match), can also
# just go into 'cmake'.
install(EXPORT  ${PROJECT_NAME}Targets
    FILE        ${PROJECT_NAME}Targets.cmake
    NAMESPACE   proteus::
    DESTINATION lib/cmake/${PROJECT_NAME}
    COMPONENT   ${PROTEUS_DEVELOPMENT_COMPONENT})

# This makes the project importable from the build directory
export(TARGETS ${PROJECT_NAME} FILE ${PROJECT_NAME}Config.cmake)

add_library(proteus::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
#enable_testing()
#add_test(UT codegen-tests) # Add codegen-specific tests?

write_basic_package_version_file(
    ${CMAKE_BINARY_DIR}/cmake/${PROJECT_NAME}ConfigVersion.cmake
    COMPATIBILITY SameMajorVersion
)
