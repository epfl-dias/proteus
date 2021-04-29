project(olap-deprecated VERSION 0.1 LANGUAGES CXX)

foreach (target
  main
  main-caches
  main-nest
  main-str
  main-reduce
  )

  add_executable(deprecated-${target} ${target}.cpp)

  target_link_libraries(deprecated-${target}
    proteus::olap
    )
endforeach (target)

include(proteus-experiment)
