add_executable(inspect_sstable
  inspect_sstable/inspect_sstable.cpp
)
target_link_libraries(inspect_sstable PRIVATE pebbledb_core)
