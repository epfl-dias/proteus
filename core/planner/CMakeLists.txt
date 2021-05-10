project(planner VERSION 0.1 LANGUAGES CXX)

file(GLOB_RECURSE planner_scala
  CONFIGURE_DEPENDS RELATIVE
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*"
  )

file(GLOB planner_scala2
  LIST_DIRECTORIES false
  CONFIGURE_DEPENDS RELATIVE
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/project/*"
  )

list(APPEND planner_scala build.sbt ${planner_scala2})

foreach (p IN LISTS planner_scala)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${p}
    ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/planner.dir/${p} COPYONLY
    )
endforeach ()

add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/proteusplanner.jar
  # --batch: do not go into interactive mode. (Otherwise failed cmake
  #          tasks wait for user input)
  COMMAND sbt --no-share --batch assembly
  COMMAND cp ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/planner.dir/target/scala-*/proteusplanner*.jar
  ${CMAKE_CURRENT_BINARY_DIR}/proteusplanner.jar
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/planner.dir
  DEPENDS ${planner_scala}
  COMMENT "Building planner jar"
  )

add_custom_target(planner ALL DEPENDS
  ${CMAKE_CURRENT_BINARY_DIR}/proteusplanner.jar
  )

# 'make install' to the correct locations (provided by GNUInstallDirs).
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/proteusplanner.jar
  DESTINATION ${CMAKE_INSTALL_BINDIR}
  COMPONENT planner_${PROTEUS_CPACK_COMP_SUFFIX_BINARIES}
  )
