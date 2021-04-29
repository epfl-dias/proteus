foreach (target
  spam-bin
  spam-csv
  spam-json
  spam-bin-csv
  spam-bin-json
  spam-csv-json
  spam-bin-csv-json
  spam-bin-csv-json-v2
  spam-csv-cached
  spam-csv-cached-columnar
  spam-json-cached
  )

  project(${target} VERSION 0.1 LANGUAGES CXX)
  add_executable(${PROJECT_NAME} ${target}.cpp)

  target_link_libraries(${PROJECT_NAME}
    proteus::olap
    )
endforeach (target)
