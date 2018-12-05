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

#ifndef RAW_STORAGE_MANAGER_HPP_
#define RAW_STORAGE_MANAGER_HPP_

#include "common/gpu/gpu-common.hpp"
#include "common/common.hpp"
#include <map>
#include <vector>

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
// //     RawStorageDescription(const RawFileDescription &desc): file_description(desc){}
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
    const void * data;
    size_t       size;
};

class StorageManager{
private:
    static std::map<std::string, std::vector<std::unique_ptr<mmap_file>>> files;
    static std::map<std::string, std::map<int, std::string> *>            dicts;
public:
    // void load  (const RawStorageDescription &desc);
    // void unload(const RawStorageDescription &desc);
    static void load            (std::string name, size_t type_size, data_loc loc);
    static void loadToGpus      (std::string name, size_t type_size);
    static void loadToCpus      (std::string name, size_t type_size);
    static void loadEverywhere  (std::string name, size_t type_size, int pref_gpu_weight = 1, int pref_cpu_weight = 1);

    static void * getDictionaryOf(std::string name);

    static void unloadAll();
    // void unload(std::string name);

    static std::vector<mem_file> getFile(std::string name);
    static std::vector<mem_file> getOrLoadFile(std::string name, size_t type_size, data_loc loc = PINNED);
};

#endif /* RAW_STORAGE_MANAGER_HPP_ */