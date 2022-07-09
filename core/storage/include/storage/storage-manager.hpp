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
#include <platform/common/common.hpp>
#include <platform/common/gpu/gpu-common.hpp>
#include <storage/mmap-file.hpp>
#include <vector>

struct mem_file {
  const void *data;
  size_t size;
};

/**
 * Object representing a pinned file and it's associated resources.
 *
 * Treat with care, as this can be a very expensive object to create/manipulate.
 * Only the loading policies and the StorageManager have to work with this
 * calls.
 *
 * To request a file from the Storage layers, see StorageManager#request
 * and StorageManager#setLoader.
 */
class FileRecord {
 public:
  std::vector<std::unique_ptr<mmap_file>> data;

  explicit FileRecord(std::vector<std::unique_ptr<mmap_file>> data);

 public:
  explicit FileRecord(std::initializer_list<std::unique_ptr<mmap_file>> data);

 public:
  FileRecord(const FileRecord &) = delete;
  FileRecord &operator=(const FileRecord &) = delete;
  FileRecord(FileRecord &&) = default;
  FileRecord &operator=(FileRecord &&) = default;
  ~FileRecord();

  static FileRecord loadToGpus(const std::string &name, size_t type_size);
  static FileRecord loadToGpus(const std::string &name, size_t type_size,
                               size_t psize, size_t offset);
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
  /**
   * Returns whether the file is currently pinned, by this file request.
   *
   * @return true if the file is pinned
   */
  [[nodiscard]] bool isPinned() const { return pinned; }

  /**
   * Force the file to be accessible until unpinned, potentially loading it if
   * needed
   */
  void pin() {
    assert(!pinned);
    pinned = true;
    data = generator();
  }

  /**
   * Allow the file to be unloaded.
   *
   * While after unpin returns the file may be unloaded, the unload is not
   * enforced and it's up to the storage subsystem to decide when to really
   * unload the file.
   */
  void unpin() {
    assert(pinned);
    pinned = false;
    data = std::shared_future<segments_t>{};
  }

  /**
   * Register the intent to pin this file soon, potentially prefetching it.
   *
   * This may allow the storage subsystem to start loading the file in the
   * background, but no guarantees are provided. The storage subsystem in
   * combination with the loading policy are free to load the file
   * asynchronously, block until the file is completely loaded, completely
   * ignore the registered intent or do anything else they deem appropriate,
   * like unloading other files to create space.
   */
  void registerIntent() { generator(); }

  /**
   * Return the number of segments for the file.
   *
   * If the file is pinned, then this should be equivalent getSegments().size(),
   * otherwise this call may be as expensive as pinning and unpinning the file.
   *
   * @return the number of segments representing the file.
   */
  size_t getSegmentCount() {
    bool was_pinned = isPinned();
    if (!was_pinned) pin();
    auto s = get().size();
    if (!was_pinned) unpin();
    return s;
  }

  /**
   * Return a reference to the file segments
   *
   * @pre the file should be pinned
   * @return vectors of segments
   */
  const segments_t &getSegments() {
    assert(isPinned());
    return get();
  }
};

namespace proteus {
class StorageLoadPolicyRegistry;
}

class StorageManager {
 private:
  std::map<std::string, std::shared_future<FileRecord>> files;
  std::map<std::string, std::map<int, std::string> *> dicts;
  std::unique_ptr<proteus::StorageLoadPolicyRegistry> loadRegistry;

 public:
  using Loader = std::function<FileRecord(StorageManager &, const std::string &,
                                          size_t typeSize)>;
  static double StorageFilePercentage;
  StorageManager();
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
      std::string name, size_t type_size, data_loc loc = FROM_REGISTRY);

 public:
  /**
   * Request a handle to a (logical) file, used to manipulate the file.
   *
   * Provides the main entry point to the storage manager, giving access
   * to manipulating files.
   *
   * NOTE: @p type_size and @p loc will be ignored by future versions and
   * should be consider unreliable until then. Use load policies to set these
   * parameters, through #setLoader and #setDefaultLoader.
   *
   * @param name        logical file name
   * @param type_size   preferred atomic word unit for opening the file
   * @param loc         preferred file loading policy
   * @return Handle that allows pinning/unpinning etc the file
   *
   * @see FileRequest
   * @see #setLoader
   * @see #setDefaultLoader
   */
  [[nodiscard]] FileRequest request(std::string name, size_t type_size,
                                    data_loc loc);
  void unloadFile(std::string name);

  /**
   * Sets a default file loader.
   * @param ld loader to use unless a specific loader is set for a given file
   * though @ref setLoader
   * @param force bool, if true unload all files to force defaultLoader change.
   *
   * @see @ref setLoader
   */
  void setDefaultLoader(Loader ld, bool force = false);
  void setLoader(std::string fileName, Loader ld, bool force = false);
  std::function<FileRecord(size_t)> getLoader(const std::string &fileName);
  void dropAllCustomLoaders();
};

#endif /* STORAGE_MANAGER_HPP_ */
