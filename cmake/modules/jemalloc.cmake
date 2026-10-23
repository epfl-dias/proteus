include(external/CMakeLists.txt.jemalloc.in)

if (CMAKE_BUILD_TYPE MATCHES Debug)
  set(PROTEUS_JEMALLOC_DEBUG "--enable-debug")
else ()
  set(PROTEUS_JEMALLOC_DEBUG "")
endif ()

# Could possibly always have this on, overheads are meant to be low
if (PROTEUS_JEMALLOC_PROF)
  set(PROTEUS_JEMALLOC_PROF "--enable-prof --enable-prof-libunwind")
else ()
  set(PROTEUS_JEMALLOC_PROF "")
endif ()

# jemalloc emits a lot of these warnings on clang, but they seem harmless
set(PROTEUS_JEMALLOC_CXXFLAGS "${CMAKE_CXX_FLAGS} -Wno-ignored-attributes -Wno-unused-command-line-argument")
set(PROTEUS_JEMALLOC_CFLAGS "${CMAKE_C_FLAGS} -Wno-ignored-attributes -Wno-unused-command-line-argument")

FetchContent_GetProperties(jemalloc)
if (NOT jemalloc_POPULATED)
  FetchContent_Populate(jemalloc)

  execute_process(COMMAND autoconf
    WORKING_DIRECTORY ${jemalloc_SOURCE_DIR}
    RESULT_VARIABLE JEMALLOC_AUTOCONF_RESULT
    )

  if (NOT JEMALLOC_AUTOCONF_RESULT EQUAL "0")
    message(FATAL_ERROR "jemalloc autoconf failed with ${JEMALLOC_AUTOCONF_RESULT}")
  endif ()

  execute_process(COMMAND ${jemalloc_SOURCE_DIR}/configure CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER}
    CFLAGS=${PROTEUS_JEMALLOC_CFLAGS} CPPFLAGS=${PROTEUS_JEMALLOC_CXXFLAGS} -C --enable-autogen
    --disable-shared --disable-libdl ${PROTEUS_JEMALLOC_DEBUG} ${PROTEUS_JEMALLOC_PROF} --prefix=${CMAKE_INSTALL_PREFIX}
    WORKING_DIRECTORY ${jemalloc_BINARY_DIR}
    RESULT_VARIABLE JEMALLOC_CONFIGURE_RESULT
    )

  if (NOT JEMALLOC_CONFIGURE_RESULT EQUAL "0")
    message(FATAL_ERROR "jemalloc configure failed with ${JEMALLOC_CONFIGURE_RESULT}")
  endif ()

  add_custom_target(make_jemalloc
    COMMAND make -j 8
    WORKING_DIRECTORY ${jemalloc_BINARY_DIR}
    BYPRODUCTS ${jemalloc_BINARY_DIR}/lib/libjemalloc.a
    )
endif ()

find_package(Threads REQUIRED)

# I _think_ if linking a executable to jemallic statically, that it should be linked first so that any other
# shared libraries will use jemalloc instead of malloc
add_library(jemalloc INTERFACE)
target_include_directories(jemalloc INTERFACE $<BUILD_INTERFACE:${jemalloc_BINARY_DIR}/include>)
target_link_libraries(jemalloc INTERFACE $<BUILD_INTERFACE:${jemalloc_BINARY_DIR}/lib/libjemalloc.a> Threads::Threads)
add_dependencies(jemalloc make_jemalloc)