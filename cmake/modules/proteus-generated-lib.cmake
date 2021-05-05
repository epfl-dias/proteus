set_target_properties(${PROJECT_NAME}
	PROPERTIES
	SOVERSION ${${PROJECT_NAME}_VERSION_MAJOR}
	VERSION ${${PROJECT_NAME}_VERSION}
	)

# Error on undefined symbols in shared libraries
set_target_properties(${PROJECT_NAME} PROPERTIES
	LINK_FLAGS "-Wl,--no-allow-shlib-undefined -Wl,-z,defs"
	)

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

set(_proteus_install_target ${PROJECT_NAME})
include(_proteus-install-lib)
