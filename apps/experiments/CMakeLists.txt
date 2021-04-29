# real-world experiments
if (REALWORLD_EXPERIMENTS)
  add_subdirectory(realworld-queries)
  add_subdirectory(realworld-symantec)
  add_subdirectory(realworld-vldb)

  # sigmod versions
  add_subdirectory(realworld)
endif ()
