get_filename_component(STANDALONE_DIR ${CMAKE_CURRENT_SOURCE_DIR} NAME)

project(${STANDALONE_DIR} VERSION 0.1 LANGUAGES CXX)

set(MAINS
    ${STANDALONE_DIR}
    )

foreach(target ${MAINS})
  add_executable(proteus${target} ${target}.cpp)
  target_link_libraries(proteus${target} codegen::olap_flags codegen::olap aeolus::engine aeolus::plugin)
  # FIXME: remove after old warnings are fixed
  target_compile_options(
        proteus${target}
      PRIVATE # Not PUBLIC to at least avoid leaking the issue across libs
        "-Wno-error=suggest-override"
        "-Wno-error=suggest-destructor-override"
  )
  install(TARGETS proteus${target}
      ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
      LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
      RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR}
      )
endforeach(target)
