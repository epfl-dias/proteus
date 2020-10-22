project(cli-flags VERSION 0.1 LANGUAGES CXX)

install(EXPORT gflags-targets DESTINATION share/gflags/cmake)

add_library(cli-flags SHARED empty.cpp)
target_link_libraries(cli-flags PUBLIC gflags::gflags)

# 'make install' to the correct locations (provided by GNUInstallDirs).
install(TARGETS cli-flags EXPORT ${PROJECT_NAME}Config.cmake
        ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})  # This is for Windows

install(EXPORT ${PROJECT_NAME}Config.cmake DESTINATION share/${PROJECT_NAME}/cmake)
export(TARGETS cli-flags gflags_shared FILE ${PROJECT_NAME}Config.cmake)

add_library(cli-flags::cli-flags ALIAS cli-flags)
