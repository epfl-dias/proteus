/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2017
        Data Intensive Applications and Systems Laboratory (DIAS)
                École Polytechnique Fédérale de Lausanne

                            All Rights Reserved.

    Permission to use, copy, modify and distribute this software and
    its documentation is hereby granted, provided that both the
    copyright notice and this permission notice appear in all copies of
    the software, derivative works or modified versions, and any
    portions thereof, and that both notices appear in supporting
    documentation.

    This code is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. THE AUTHORS
    DISCLAIM ANY LIABILITY OF ANY KIND FOR ANY DAMAGES WHATSOEVER
    RESULTING FROM THE USE OF THIS SOFTWARE.
*/

#ifndef STORAGE_MANAGER_HPP_
#define STORAGE_MANAGER_HPP_

#include <future>
#include <map>
#include <vector>

#include "common/common.hpp"
#include "common/gpu/gpu-common.hpp"
#include "mmap-file.hpp"

// class StorageManager;

// class RawFileDescription{
// private:
//     std::string name;
// public:
//     RawFileDescription(std::string name): name(name){}
// };

// // class RawStorageDescription{
// // private:
// //     const RawFileDescription file_description;

// // public:
// //     RawStorageDescription(const RawFileDescription &desc):
// file_description(desc){}
// // };

// class RawFile{
// private:

// public:
//     void open();
//     void * next();
//     void close();

// friend StorageManager;
// };

struct mem_file {
  const void *data;
  size_t size;
};

class FileRecord {
 public:
  std::vector<std::unique_ptr<mmap_file>> data;

 private:
  explicit FileRecord(std::vector<std::unique_ptr<mmap_file>> data);
  explicit FileRecord(std::initializer_list<std::unique_ptr<mmap_file>> data);

 public:
  FileRecord(const FileRecord &) = delete;
  FileRecord &operator=(const FileRecord &) = delete;
  FileRecord(FileRecord &&) = default;
  FileRecord &operator=(FileRecord &&) = default;
  ~FileRecord();

  static FileRecord loadToGpus(const std::string &name, size_t type_size);
  static FileRecord loadDistributed(const std::string &name, size_t type_size);
  static FileRecord loadToCpus(const std::string &name, size_t type_size);
  static FileRecord loadToCpus(const std::string &name, size_t type_size,
                               size_t psize, size_t offset);
  static FileRecord loadEverywhere(const std::string &name, size_t type_size,
                                   int pref_gpu_weight, int pref_cpu_weight);

  static FileRecord load(const std::string &name, size_t type_size,
                         data_loc loc);
};

class FileRequest {
 public:
  typedef std::vector<mem_file> segments_t;

 private:
  std::function<std::future<segments_t>()> generator;
  std::shared_future<segments_t> data;
  bool pinned;

  const std::vector<mem_file> &get() { return data.get(); }

 private:
  explicit FileRequest(decltype(generator) generator)
      : generator(std::move(generator)), pinned(false) {}

  friend class StorageManager;

 public:
  [[nodiscard]] bool isPinned() const { return pinned; }

  void pin() {
    assert(!pinned);
    pinned = true;
    data = generator();
  }

  void unpin() {
    assert(pinned);
    pinned = false;
    data = std::shared_future<segments_t>{};
  }

  void registerIntent() { generator(); }

  size_t getSegmentCount() {
    bool was_pinned = isPinned();
    if (!was_pinned) pin();
    auto s = get().size();
    if (!was_pinned) unpin();
    return s;
  }

  const segments_t &getSegments() {
    assert(isPinned());
    return get();
  }
};

class StorageManager {
 private:
  std::map<std::string, std::shared_future<FileRecord>> files;
  std::map<std::string, std::map<int, std::string> *> dicts;

 public:
  ~StorageManager();

  static StorageManager &getInstance();
  // void load  (const RawStorageDescription &desc);
  // void unload(const RawStorageDescription &desc);
  void load(std::string name, size_t type_size, data_loc loc);
  void loadToGpus(std::string name, size_t type_size);
  void loadToCpus(std::string name, size_t type_size);
  void loadEverywhere(std::string name, size_t type_size,
                      int pref_gpu_weight = 1, int pref_cpu_weight = 1);

  void *getDictionaryOf(std::string name);

  void unloadAll();
  // void unload(std::string name);

  [[nodiscard]] std::future<std::vector<mem_file>> getFile(std::string name);
  [[nodiscard]] std::future<std::vector<mem_file>> getOrLoadFile(
      std::string name, size_t type_size, data_loc loc = PINNED);
  [[nodiscard]] FileRequest request(std::string name, size_t type_size,
                                    data_loc loc);
  void unloadFile(std::string name);
};

#endif /* STORAGE_MANAGER_HPP_ */
