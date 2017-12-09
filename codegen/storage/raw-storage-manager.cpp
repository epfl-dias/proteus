/*
    RAW -- High-performance querying over raw, never-seen-before data.

                            Copyright (c) 2017
        Data Intensive Applications and Systems Labaratory (DIAS)
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

#include "storage/raw-storage-manager.hpp"
#include <algorithm>

std::map<std::string, std::vector<std::unique_ptr<mmap_file>>> StorageManager::files;

void StorageManager::load(std::string name, data_loc loc){
    time_block t("Topen (" + name + "): ");

    auto it = files.emplace(name, std::vector<std::unique_ptr<mmap_file>>{});
    assert(it.second && "File already loaded!");
    it.first->second.emplace_back(new mmap_file(name, loc));
}

void StorageManager::loadToGpus(std::string name){
    time_block t("Topen (" + name + "): ");

    int devices = get_num_of_gpus();

    auto it = files.emplace(name, std::vector<std::unique_ptr<mmap_file>>{});
    assert(it.second && "File already loaded!");

    size_t filesize  = ::getFileSize(name.c_str());

    size_t pack_alignment = sysconf(_SC_PAGE_SIZE); //required by mmap
    pack_alignment = std::max(pack_alignment, (size_t) h_vector_size * 4);

    size_t part_size = (((filesize + pack_alignment - 1)/pack_alignment + devices - 1) / devices) * pack_alignment; //FIXME: assumes maximum record size of 128Bytes

    for (int d = 0 ; d < devices ; ++d){
        if (part_size * d < filesize){
            set_device_on_scope cd(d);
            it.first->second.emplace_back(new mmap_file(name, GPU_RESIDENT, std::min(part_size, filesize - part_size * d), part_size * d));
        }
    }
}

void StorageManager::loadToCpus(std::string name){
    time_block t("Topen (" + name + "): ");

#ifndef NNUMA
    int devices = numa_num_task_nodes();
#else
    int devices = 1;
#endif

    auto it = files.emplace(name, std::vector<std::unique_ptr<mmap_file>>{});
    assert(it.second && "File already loaded!");

    size_t filesize  = ::getFileSize(name.c_str());

    size_t pack_alignment = sysconf(_SC_PAGE_SIZE); //required by mmap
    pack_alignment = std::max(pack_alignment, (size_t) h_vector_size * 4);

    size_t part_size = (((filesize + pack_alignment - 1)/pack_alignment + devices - 1) / devices) * pack_alignment; //FIXME: assumes maximum record size of 128Bytes

    for (int d = 0 ; d < devices ; ++d){
        if (part_size * d < filesize){
            set_exec_location_on_scope cd(cpu_numa_affinity[d]);
            it.first->second.emplace_back(new mmap_file(name, PINNED, std::min(part_size, filesize - part_size * d), part_size * d));
        }
    }
}

void StorageManager::loadEverywhere(std::string name, int pref_gpu_weight, int pref_cpu_weight){
    time_block t("Topen (" + name + "): ");

#ifndef NNUMA
    int devices_sock = numa_num_task_nodes();
#else
    int devices_sock = 1;
#endif
    int devices_gpus = get_num_of_gpus    ();
    int devices      = devices_sock * pref_cpu_weight + devices_gpus * pref_gpu_weight;

    auto it = files.emplace(name, std::vector<std::unique_ptr<mmap_file>>{});
    assert(it.second && "File already loaded!");

    size_t filesize  = ::getFileSize(name.c_str());

    size_t pack_alignment = sysconf(_SC_PAGE_SIZE); //required by mmap
    pack_alignment = std::max(pack_alignment, (size_t) h_vector_size * 4);

    size_t part_size = (((filesize + pack_alignment - 1)/pack_alignment + devices - 1) / devices) * pack_alignment; //FIXME: assumes maximum record size of 128Bytes

    for (int d = 0 ; d < devices ; ++d){
        if (part_size * d < filesize){
            if (d < devices_sock * pref_cpu_weight){
                set_exec_location_on_scope cd(cpu_numa_affinity[d % devices_sock]);
                it.first->second.emplace_back(new mmap_file(name, PINNED      , std::min(part_size, filesize - part_size * d), part_size * d));
            } else {
                set_device_on_scope cd((d - devices_sock * pref_cpu_weight) % devices_gpus);
                it.first->second.emplace_back(new mmap_file(name, GPU_RESIDENT, std::min(part_size, filesize - part_size * d), part_size * d));
            }
        }
    }
}

void StorageManager::unloadAll(){
    files.clear();
}

std::vector<mem_file> StorageManager::getFile(std::string name){
    assert(files.count(name) > 0 && "File not loaded!");
    const auto &f = files[name];
    std::vector<mem_file> mfiles{f.size()};
    for (size_t i = 0 ; i < mfiles.size() ; ++i) {
        mfiles[i].data = f[i]->getData    ();
        mfiles[i].size = f[i]->getFileSize();
    }
    return mfiles;
}

std::vector<mem_file> StorageManager::getOrLoadFile(std::string name, data_loc loc){
    if (files.count(name) == 0){
        load(name, loc);
    }
    return getFile(name);
}