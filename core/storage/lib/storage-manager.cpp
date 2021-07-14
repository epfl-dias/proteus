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

#include <algorithm>
#include <fstream>
#include <magic_enum.hpp>
#include <platform/memory/block-manager.hpp>
#include <platform/network/infiniband/infiniband-manager.hpp>
#include <platform/threadpool/threadpool.hpp>
#include <platform/topology/affinity_manager.hpp>
#include <platform/topology/topology.hpp>
#include <platform/util/timing.hpp>
#include <storage/mmap-file.hpp>
#include <storage/storage-manager.hpp>

#include "storage-load-policy-registry.hpp"

StorageManager &StorageManager::getInstance() {
  static StorageManager sm;
  return sm;
}

StorageManager::~StorageManager() { assert(files.empty()); }

FileRecord::FileRecord(std::vector<std::unique_ptr<mmap_file>> data)
    : data(std::move(data)) {
  for (const auto &s : this->data) {
    InfiniBandManager::reg(s->getData(), s->getFileSize());
  }
}

FileRecord::~FileRecord() {
  for (const auto &s : this->data) {
    InfiniBandManager::unreg(s->getData());
  }
}

FileRecord FileRecord::load(const std::string &name, size_t type_size,
                            data_loc loc) {
  if (loc == ALLSOCKETS) return loadToCpus(name, type_size);

  if (loc == ALLGPUS) return loadToGpus(name, type_size);

  if (loc == EVERYWHERE) return loadEverywhere(name, type_size, 1, 1);

  if (loc == DISTRIBUTED) return loadDistributed(name, type_size);

  if (loc == FROM_REGISTRY) {
    return StorageManager::getInstance().getLoader(name)(type_size);
  }

  time_block t("Topen (" + name + "): ",
               TimeRegistry::Key{"Data loading (current)"});

  decltype(data) p;
  p.emplace_back(std::make_unique<mmap_file>(name, loc));
  return FileRecord{std::move(p)};
}

FileRecord FileRecord::loadDistributed(const std::string &name,
                                       size_t type_size) {
  time_block t("Topen distributed (" + name + "): ");
  size_t factor = type_size / sizeof(int32_t);

  auto devices = InfiniBandManager::server_count();

  size_t filesize = ::getFileSize(name.c_str()) / factor;

  size_t pack_alignment = sysconf(_SC_PAGE_SIZE);  // required by mmap
  // in order to do that without the schema, we have to take the worst case
  // of a file with a single-byte column and a 64bit column and align based
  // on that. Otherwise, the segments may be misaligned
  pack_alignment = std::max(pack_alignment, BlockManager::block_size);

  size_t part_size =
      (((filesize + pack_alignment - 1) / pack_alignment + devices - 1) /
       devices) *
      pack_alignment;  // FIXME: assumes maximum record size of 128Bytes

  size_t d = InfiniBandManager::server_id();

  // Protect from the underflow:
  size_t rem = part_size * d < filesize ? filesize - part_size * d : 0;
  return loadToCpus(name, type_size, std::min(part_size, rem) * factor,
                    part_size * d * factor);
}

FileRecord FileRecord::loadToGpus(const std::string &name, size_t type_size) {
  return loadToGpus(name, type_size, ::getFileSize(name.c_str()), 0);
}

FileRecord FileRecord::loadToGpus(const std::string &name, size_t type_size,
                                  size_t psize, size_t offset) {
  time_block t("Topen (" + name + "): ",
               TimeRegistry::Key{"Data loading (GPUs)"});
  const auto &topo = topology::getInstance();

  size_t factor = type_size / sizeof(int32_t);

  auto devices = topo.getGpuCount();

  size_t filesize = psize / factor;

  size_t pack_alignment = sysconf(_SC_PAGE_SIZE);  // required by mmap
  // in order to do that without the schema, we have to take the worst case
  // of a file with a single-byte column and a 64bit column and align based
  // on that. Otherwise, the segments may be misaligned
  pack_alignment = std::max(pack_alignment, BlockManager::block_size);

  size_t part_size =
      (((filesize + pack_alignment - 1) / pack_alignment + devices - 1) /
       devices) *
      pack_alignment;  // FIXME: assumes maximum record size of 128Bytes

  std::vector<std::unique_ptr<mmap_file>> partitions;
  partitions.reserve(devices);
  int d = 0;
  for (const auto &gpu : topo.getGpus()) {
    if (part_size * d < filesize) {
      set_device_on_scope cd(gpu);
      partitions.emplace_back(std::make_unique<mmap_file>(
          name, GPU_RESIDENT,
          std::min(part_size, filesize - part_size * d) * factor,
          part_size * d * factor + offset));
    }
    ++d;
  }
  return FileRecord{std::move(partitions)};
}

FileRecord FileRecord::loadToCpus(const std::string &name, size_t type_size) {
  return loadToCpus(name, type_size, ::getFileSize(name.c_str()), 0);
}

FileRecord FileRecord::loadToCpus(const std::string &name, size_t type_size,
                                  size_t psize, size_t offset) {
  time_block t("Topen (" + name + "): ",
               TimeRegistry::Key{"Data loading (CPUs)"});
  const auto &topo = topology::getInstance();

  size_t factor = type_size / sizeof(int32_t);

  auto devices = topo.getCpuNumaNodeCount();

  size_t filesize = psize / factor;

  size_t pack_alignment = sysconf(_SC_PAGE_SIZE);  // required by mmap
  // in order to do that without the schema, we have to take the worst case
  // of a file with a single-byte column and a 64bit column and align based
  // on that. Otherwise, the segments may be misaligned
  pack_alignment = std::max(pack_alignment, BlockManager::block_size);

  size_t part_size =
      (((filesize + pack_alignment - 1) / pack_alignment + devices - 1) /
       devices) *
      pack_alignment;  // FIXME: assumes maximum record size of 128Bytes

  decltype(data) partitions;
  partitions.reserve(devices);
  int d = 0;
  for (const auto &cpu : topo.getCpuNumaNodes()) {
    if (part_size * d < filesize) {
      set_exec_location_on_scope cd(cpu);
      partitions.emplace_back(std::make_unique<mmap_file>(
          name, PINNED, std::min(part_size, filesize - part_size * d) * factor,
          part_size * d * factor + offset));
    }
    ++d;
  }
  return FileRecord{std::move(partitions)};
}

FileRecord FileRecord::loadEverywhere(const std::string &name, size_t type_size,
                                      int pref_gpu_weight,
                                      int pref_cpu_weight) {
  time_block t("Topen (" + name + "): ",
               TimeRegistry::Key{"Data loading (everywhere)"});
  const auto &topo = topology::getInstance();

  size_t factor = type_size / sizeof(int32_t);

  auto devices_gpus = topo.getGpuCount();
  auto devices_sock = topo.getCpuNumaNodeCount();
  int devices = devices_sock * pref_cpu_weight + devices_gpus * pref_gpu_weight;

  size_t filesize = ::getFileSize(name.c_str());

  size_t pack_alignment = sysconf(_SC_PAGE_SIZE);  // required by mmap
  // in order to do that without the schema, we have to take the worst case
  // of a file with a single-byte column and a 64bit column and align based
  // on that. Otherwise, the segments may be misaligned
  pack_alignment = std::max(pack_alignment, BlockManager::block_size);

  size_t part_size =
      (((filesize + pack_alignment - 1) / pack_alignment + devices - 1) /
       devices) *
      pack_alignment;  // FIXME: assumes maximum record size of 128Bytes

  const auto &gpu_vec = topo.getGpus();
  const auto &sck_vec = topo.getCpuNumaNodes();

  decltype(data) partitions;
  partitions.reserve(devices);
  for (int d = 0; d < devices; ++d) {
    if (part_size * d < filesize) {
      if (d < devices_sock * pref_cpu_weight) {
        set_exec_location_on_scope cd(sck_vec[d % devices_sock]);
        partitions.emplace_back(std::make_unique<mmap_file>(
            name, PINNED,
            std::min(part_size, filesize - part_size * d) * factor,
            part_size * d * factor));
      } else {
        set_device_on_scope cd(
            gpu_vec[(d - devices_sock * pref_cpu_weight) % devices_gpus]);
        partitions.emplace_back(std::make_unique<mmap_file>(
            name, GPU_RESIDENT,
            std::min(part_size, filesize - part_size * d) * factor,
            part_size * d * factor));
      }
    }
  }
  return FileRecord{std::move(partitions)};
}

void StorageManager::load(std::string name, size_t type_size, data_loc loc) {
  files.emplace(name,
                ThreadPool::getInstance().enqueue([=, aff = exec_location{}]() {
                  set_exec_location_on_scope e{aff};
                  return FileRecord::load(name, type_size, loc);
                }));
}

void StorageManager::loadToGpus(std::string name, size_t type_size) {
  load(name, type_size, ALLGPUS);
}

void StorageManager::loadToCpus(std::string name, size_t type_size) {
  load(name, type_size, ALLSOCKETS);
}

void StorageManager::loadEverywhere(std::string name, size_t type_size,
                                    int pref_gpu_weight, int pref_cpu_weight) {
  files.emplace(name,
                ThreadPool::getInstance().enqueue([=, aff = exec_location{}]() {
                  set_exec_location_on_scope e{aff};
                  return FileRecord::loadEverywhere(
                      name, type_size, pref_gpu_weight, pref_cpu_weight);
                }));
}

void StorageManager::unloadAll() { files.clear(); }

void StorageManager::unloadFile(std::string name) {
  if (files.count(name) == 0) return;
  files.erase(name);
}

std::future<std::vector<mem_file>> StorageManager::getFile(std::string name) {
  if (files.count(name) == 0) {
    LOG(ERROR) << "File " << name << " not loaded";
  }
  assert(files.count(name) > 0 && "File not loaded!");
  return ThreadPool::getInstance().enqueue(
      [](auto ffut) {
        const auto &f = ffut.get().data;
        std::vector<mem_file> mfiles;
        mfiles.reserve(f.size());
        for (const auto &fi : f) {
          mfiles.emplace_back(mem_file::fromLocal(*fi));
        }
        return mfiles;
      },
      files[name]);
}

std::future<std::vector<mem_file>> StorageManager::getOrLoadFile(
    std::string name, size_t type_size, data_loc loc) {
  if (files.count(name) == 0) {
    LOG(INFO) << "File " << name << " not loaded, loading it to "
              << magic_enum::enum_name(loc);
    load(name, type_size, loc);
  }
  return getFile(name);
}

FileRequest StorageManager::request(std::string name, size_t type_size,
                                    data_loc loc) {
  return FileRequest{[=, this, aff = exec_location{}]() {
    set_exec_location_on_scope e{aff};
    return this->getOrLoadFile(name, type_size, loc);
  }};
}

void *StorageManager::getDictionaryOf(std::string name) {
  if (dicts.count(name) == 0) {
    std::ifstream dictfile(name + ".dict");

    auto *d = new std::map<int, std::string>;

    std::string line;
    while (std::getline(dictfile, line)) {
      size_t index = line.find_last_of(':');
      assert(index != std::string::npos && "Invalid file");

      int encoding = std::stoi(line.substr(index + 1, line.size() - 1));
      d->emplace(encoding, line.substr(0, index));
    }

    dicts[name] = d;
  }

  return dicts[name];
}

StorageManager::StorageManager()
    : loadRegistry(std::make_unique<proteus::StorageLoadPolicyRegistry>(
          [](StorageManager &stManager, const std::string &fileName,
             size_t typeSize) {
            return FileRecord::load(fileName, typeSize, ALLSOCKETS);
          })) {}

void StorageManager::setDefaultLoader(Loader ld, bool force) {
  if (force) {
    // Unload all so that next time we use the new policy
    unloadAll();
  }

  loadRegistry->setDefaultLoader(std::move(ld));
}

void StorageManager::setLoader(std::string fileName, Loader ld, bool force) {
  if (force) {
    // Remove file from loaded files, to force loading with new policy next time
    files.erase(fileName);
  }
  loadRegistry->setLoader(std::move(fileName), std::move(ld));
}

void StorageManager::dropAllCustomLoaders() {
  loadRegistry->dropAllCustomLoaders();
}

std::function<FileRecord(size_t)> StorageManager::getLoader(
    const std::string &fileName) {
  return [this, ld = loadRegistry->at(fileName),
          f = fileName](size_t typeSize) { return ld(*this, f, typeSize); };
}
