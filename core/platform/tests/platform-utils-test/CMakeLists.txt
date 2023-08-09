# Set the name of the app from the folder name
get_filename_component(TS_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)

add_proteus_unit_test(unit-${TS_NAME} proteus::platform atomic-bit-set-test.cpp)

target_link_libraries(unit-${TS_NAME}
  PUBLIC
  proteus::platform
  proteus::platform-test-main
  )
