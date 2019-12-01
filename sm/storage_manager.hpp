/*
                  AEOLUS - In-Memory HTAP-Ready OLTP Engine

                              Copyright (c) 2019-2019
           Data Intensive Applications and Systems Laboratory (DIAS)
                   Ecole Polytechnique Federale de Lausanne

                              All Rights Reserved.

      Permission to use, copy, modify and distribute this software and its
    documentation is hereby granted, provided that both the copyright notice
  and this permission notice appear in all copies of the software, derivative
  works or modified versions, and any portions thereof, and that both notices
                      appear in supporting documentation.

  This code is distributed in the hope that it will be useful, but WITHOUT ANY
 WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 A PARTICULAR PURPOSE. THE AUTHORS AND ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE
DISCLAIM ANY LIABILITY OF ANY KIND FOR ANY DAMAGES WHATSOEVER RESULTING FROM THE
                             USE OF THIS SOFTWARE.
*/

#ifndef STORAGE_MANAGER_HPP_
#define STORAGE_MANAGER_HPP_

#include <numa.h>
#include <numaif.h>

#include <iostream>
#include <map>
#include <string>

namespace storage {

struct mem_file {
  std::string path;
  size_t size;
  size_t unit_size;
  uint64_t num_records;
  uint32_t snapshot_epoch;  // for master-version based snapshotting
};

class StorageManager {
 protected:
 public:
  // Singleton
  static StorageManager &getInstance() {
    static StorageManager instance;
    return instance;
  }

  // Prevent copies
  StorageManager(const StorageManager &) = delete;
  void operator=(const StorageManager &) = delete;

  StorageManager(StorageManager &&) = delete;
  StorageManager &operator=(StorageManager &&) = delete;

  void init();
  void shutdown();

  void snapshot();
  bool alloc_shm(const std::string &key, const size_t size_bytes,
                 const size_t unit_size);
  bool remove_shm(const std::string &key);

 private:
  std::map<std::string, struct mem_file> mappings;

  StorageManager() {}
  ~StorageManager() {}
};

}  // namespace storage

#endif /* STORAGE_MANAGER_HPP_ */
