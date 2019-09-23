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

#include "storage/storage-manager.hpp"

#include <algorithm>
#include <fstream>

#include "memory/block-manager.hpp"
#include "topology/affinity_manager.hpp"
#include "topology/topology.hpp"
#include "util/timing.hpp"

std::map<std::string, std::vector<std::unique_ptr<mmap_file>>>
    StorageManager::files;
std::map<std::string, std::map<int, std::string> *> StorageManager::dicts;

void StorageManager::load(std::string name, size_t type_size, data_loc loc) {
  if (loc == ALLSOCKETS) {
    loadToCpus(name, type_size);
    return;
  }

  if (loc == ALLGPUS) {
    loadToGpus(name, type_size);
    return;
  }

  if (loc == EVERYWHERE) {
    loadEverywhere(name, type_size, 1, 1);
    return;
  }

  time_block t("Topen (" + name + "): ");

  auto it = files.emplace(name, std::vector<std::unique_ptr<mmap_file>>{});
  assert(it.second && "File already loaded!");
  it.first->second.emplace_back(new mmap_file(name, loc));
}

void StorageManager::loadToGpus(std::string name, size_t type_size) {
  time_block t("Topen (" + name + "): ");
  const auto &topo = topology::getInstance();

  size_t factor = type_size / sizeof(int32_t);

  auto devices = topo.getGpuCount();

  auto it = files.emplace(name, std::vector<std::unique_ptr<mmap_file>>{});
  assert(it.second && "File already loaded!");

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

  int d = 0;
  for (const auto &gpu : topo.getGpus()) {
    if (part_size * d < filesize) {
      set_device_on_scope cd(gpu);
      it.first->second.emplace_back(
          new mmap_file(name, GPU_RESIDENT,
                        std::min(part_size, filesize - part_size * d) * factor,
                        part_size * d * factor));
    }
    ++d;
  }
}

void StorageManager::loadToCpus(std::string name, size_t type_size) {
  time_block t("Topen (" + name + "): ");
  const auto &topo = topology::getInstance();

  size_t factor = type_size / sizeof(int32_t);

  auto devices = topo.getCpuNumaNodeCount();

  auto it = files.emplace(name, std::vector<std::unique_ptr<mmap_file>>{});
  assert(it.second && "File already loaded!");

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

  int d = 0;
  for (const auto &cpu : topo.getCpuNumaNodes()) {
    if (part_size * d < filesize) {
      set_exec_location_on_scope cd(cpu);
      it.first->second.emplace_back(new mmap_file(
          name, PINNED, std::min(part_size, filesize - part_size * d) * factor,
          part_size * d * factor));
    }
    ++d;
  }
}

void StorageManager::loadEverywhere(std::string name, size_t type_size,
                                    int pref_gpu_weight, int pref_cpu_weight) {
  time_block t("Topen (" + name + "): ");
  const auto &topo = topology::getInstance();

  size_t factor = type_size / sizeof(int32_t);

  auto devices_gpus = topo.getGpuCount();
  auto devices_sock = topo.getCpuNumaNodeCount();
  int devices = devices_sock * pref_cpu_weight + devices_gpus * pref_gpu_weight;

  auto it = files.emplace(name, std::vector<std::unique_ptr<mmap_file>>{});
  assert(it.second && "File already loaded!");

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

  for (int d = 0; d < devices; ++d) {
    if (part_size * d < filesize) {
      if (d < devices_sock * pref_cpu_weight) {
        set_exec_location_on_scope cd(sck_vec[d % devices_sock]);
        it.first->second.emplace_back(new mmap_file(
            name, PINNED,
            std::min(part_size, filesize - part_size * d) * factor,
            part_size * d * factor));
      } else {
        set_device_on_scope cd(
            gpu_vec[(d - devices_sock * pref_cpu_weight) % devices_gpus]);
        it.first->second.emplace_back(new mmap_file(
            name, GPU_RESIDENT,
            std::min(part_size, filesize - part_size * d) * factor,
            part_size * d * factor));
      }
    }
  }
}

void StorageManager::unloadAll() { files.clear(); }

void StorageManager::unloadFile(std::string name) {
  if (files.count(name) == 0) {
    LOG(ERROR) << "File " << name << " not loaded";
  }

  assert(files.count(name) > 0 && "File not loaded!");

  files.erase(name);
}

std::vector<mem_file> StorageManager::getFile(std::string name) {
  if (files.count(name) == 0) {
    LOG(ERROR) << "File " << name << " not loaded";
  }
  assert(files.count(name) > 0 && "File not loaded!");
  const auto &f = files[name];
  std::vector<mem_file> mfiles{f.size()};
  for (size_t i = 0; i < mfiles.size(); ++i) {
    mfiles[i].data = f[i]->getData();
    mfiles[i].size = f[i]->getFileSize();
  }
  return mfiles;
}

std::vector<mem_file> StorageManager::getOrLoadFile(std::string name,
                                                    size_t type_size,
                                                    data_loc loc) {
  if (files.count(name) == 0) {
    LOG(INFO) << "File " << name << " not loaded, loading it to " << loc;
    std::cout << "File " << name << " not loaded, loading it to " << loc
              << std::endl;
    load(name, type_size, loc);
  } else {
    LOG(INFO) << "Using loaded version of file " << name;
    std::cout << "Using loaded version of file " << name << std::endl;
  }
  return getFile(name);
}

void *StorageManager::getDictionaryOf(std::string name) {
  if (dicts.count(name) == 0) {
    std::ifstream dictfile(name + ".dict");

    std::map<int, std::string> *d = new std::map<int, std::string>;

    std::string line;
    while (std::getline(dictfile, line)) {
      size_t index = line.find_last_of(":");
      assert(index != std::string::npos && "Invalid file");

      int encoding = std::stoi(line.substr(index + 1, line.size() - 1));
      d->emplace(encoding, line.substr(0, index));
    }

    dicts[name] = d;
  }

  return dicts[name];
}
