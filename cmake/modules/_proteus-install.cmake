# Private module, for code factorisation
# _proteus_install_target: Target name
# _proteus_install_dev: Development component name
# _proteus_install_bin: Binary component name

install(TARGETS ${_proteus_install_target}
	ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
		COMPONENT          ${_proteus_install_dev}
	LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
		COMPONENT          ${_proteus_install_bin}
		NAMELINK_COMPONENT ${_proteus_install_dev}
	RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} # This is for Windows
		COMPONENT          ${_proteus_install_bin}
)
