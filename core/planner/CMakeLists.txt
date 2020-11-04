project(clotho VERSION 0.1 LANGUAGES CXX)

file(GLOB_RECURSE clotho_scala
	CONFIGURE_DEPENDS RELATIVE
		"${CMAKE_CURRENT_SOURCE_DIR}"
		"${CMAKE_CURRENT_SOURCE_DIR}/src/*"
)

file(GLOB clotho_scala2
	LIST_DIRECTORIES false
	CONFIGURE_DEPENDS RELATIVE
		"${CMAKE_CURRENT_SOURCE_DIR}"
		"${CMAKE_CURRENT_SOURCE_DIR}/project/*"
)

list(APPEND clotho_scala build.sbt ${clotho_scala2})

foreach(p IN LISTS clotho_scala)
	configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${p}
		${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/clotho.dir/${p} COPYONLY
	)
endforeach()

add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/clotho.jar
	# --batch: do not go into interactive mode. (Otherwise failed cmake
	#          tasks wait for user input)
	COMMAND sbt --no-share --batch assembly
	COMMAND cp ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/clotho.dir/target/scala-*/clotho-*.jar
		${CMAKE_CURRENT_BINARY_DIR}/clotho.jar
	WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/clotho.dir
	DEPENDS ${clotho_scala}
	COMMENT "Building clotho.jar"
)

add_custom_target(clotho ALL DEPENDS
	${CMAKE_CURRENT_BINARY_DIR}/clotho.jar
)

# 'make install' to the correct locations (provided by GNUInstallDirs).
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/clotho.jar
	DESTINATION ${CMAKE_INSTALL_BINDIR}
	COMPONENT planner_${PROTEUS_CPACK_COMP_SUFFIX_BINARIES}
)
