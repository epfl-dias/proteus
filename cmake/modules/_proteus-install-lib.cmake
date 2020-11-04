# Private module, for code factorisation
# _proteus_install_target: Target name
# _proteus_install_dev: Development component name
# _proteus_install_bin: Binary component name

set(_proteus_install_target ${PROJECT_NAME})

#set(_proteus_install_dev ${PROJECT_NAME}_Development)
#set(_proteus_install_bin ${PROJECT_NAME}_RunTime)
set(_proteus_install_dev ${PROJECT_NAME}_${PROTEUS_CPACK_COMP_SUFFIX_DEV})
set(_proteus_install_bin ${PROJECT_NAME}_${PROTEUS_CPACK_COMP_SUFFIX_BINARIES})


install(TARGETS ${_proteus_install_target}
	EXPORT ${_proteus_install_target}Targets
	ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
		COMPONENT          ${_proteus_install_dev}
	LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
		COMPONENT          ${_proteus_install_bin}
		NAMELINK_COMPONENT ${_proteus_install_dev}
	RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} # This is for Windows
		COMPONENT          ${_proteus_install_bin}
# Do not use PUBLIC_HEADER as it can't deduce the public header structure
#    set_target_properties(olap PROPERTIES PUBLIC_HEADER ${olap_public_hxx})
#	PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
	INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Allow libraries without public headers to also use this module
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/include/")
	install(DIRECTORY  "${CMAKE_CURRENT_SOURCE_DIR}/include/"
		DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
		COMPONENT   ${_proteus_install_dev}
	)
endif()

# This makes the project importable from the install directory
# Put config file in per-project dir (name MUST match), can also
# just go into 'cmake'.
install(EXPORT      ${PROJECT_NAME}Targets
	FILE        ${PROJECT_NAME}Targets.cmake
	NAMESPACE   ${PROJECT_NAME}::
	DESTINATION lib/cmake/${PROJECT_NAME}
	COMPONENT   ${_proteus_install_dev}
)

# This makes the project importable from the build directory
export(TARGETS ${PROJECT_NAME} FILE ${PROJECT_NAME}Config.cmake)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
	${CMAKE_BINARY_DIR}/cmake/${PROJECT_NAME}ConfigVersion.cmake
	COMPATIBILITY SameMajorVersion
)
