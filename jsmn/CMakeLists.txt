project(jsmn VERSION 0.1 LANGUAGES CXX)

set(jsmn_src
		jsmn.c
	)
add_library(jsmn SHARED ${jsmn_src})

# Define headers for this library. PUBLIC headers are used for
# compiling the library, and will be added to consumers' build
# paths.
target_include_directories(jsmn PUBLIC
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
		$<INSTALL_INTERFACE:.>
		PRIVATE src)

# 'make install' to the correct locations (provided by GNUInstallDirs).
install(TARGETS jsmn EXPORT ${PROJECT_NAME}Config
		ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
		LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
		RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})  # This is for Windows

# This makes the project importable from the install directory
# Put config file in per-project dir (name MUST match), can also
# just go into 'cmake'.
install(EXPORT ${PROJECT_NAME}Config DESTINATION share/${PROJECT_NAME}/cmake)

# This makes the project importable from the build directory
export(TARGETS jsmn FILE ${PROJECT_NAME}Config.cmake)
