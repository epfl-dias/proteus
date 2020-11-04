project(symantec-json VERSION 0.1 LANGUAGES CXX)

add_executable(${PROJECT_NAME} ${PROJECT_NAME}.cpp)

target_link_libraries(${PROJECT_NAME}
        proteus::olap

include(proteus-experiment)
