# Intel VTune profile support

find_library(VTUNE ittnotify HINTS
		"/opt/intel/vtune_amplifier/lib64"
		"~/intel/vtune_amplifier/lib64"
		"/opt/intel/vtune_profiler/lib64"
		"~/intel/vtune_profiler/lib64"
		"/opt/intel/oneapi/vtune/latest/lib64")
if(VTUNE AND VTUNE_ENABLE)
	get_filename_component(VTUNE_LIBRARY_DIR ${VTUNE} DIRECTORY)
	get_filename_component(VTUNE_ROOT ${VTUNE_LIBRARY_DIR} DIRECTORY)
	add_library(vtune::vtune INTERFACE IMPORTED)
	set_target_properties(vtune::vtune PROPERTIES
			INTERFACE_INCLUDE_DIRECTORIES "${VTUNE_ROOT}/include"
			INTERFACE_LINK_LIBRARIES "${VTUNE}"
			)
	message(STATUS "Vtune root: ${VTUNE_ROOT}")
	message(STATUS "Vtune lib: ${VTUNE}")
	message(STATUS "Vtune include: ${VTUNE_ROOT}/include")
endif()
