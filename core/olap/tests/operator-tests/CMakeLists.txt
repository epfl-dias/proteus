# Set the name of the app from the folder name
get_filename_component(TS_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)

add_proteus_unit_test(unit-${TS_NAME} proteus::olap tests-join-sailors.cpp)

target_link_libraries(unit-${TS_NAME}
  PUBLIC
  proteus::olap-test
  proteus::olap-test-utils
  proteus::olap-test-main
  )
