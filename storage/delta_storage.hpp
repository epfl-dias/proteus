/*                             Copyright (c) 2019-2019
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

#ifndef DELTA_STORAGE_HPP_
#define DELTA_STORAGE_HPP_

#include <iostream>
#include "glo.hpp"
//#include "indexes/hash_index.hpp"
#include "storage/memory_manager.hpp"
//#include "transactions/transaction_manager.hpp"
namespace storage {
/*
    TODO:
      - Memory management sucks
      - Delta size is not resizeable
      - same locl for all 3 types of data-struc, parition and have fine-grained
          locking
      - templated so that it can have multiple types of indexing
*/

/* Currently DeltaStore is not resizeable*/
class DeltaStore {
 public:
  DeltaStore(size_t rec_size, uint64_t initial_num_objs) {
    // std::unique_lock<std::mutex> lock(this->glbl_lck);

    total_ver_capacity = initial_num_objs * global_conf::max_ver_factor;
    total_tuple_capacity = initial_num_objs;

    size_t ver_list_mem_req =
        sizeof(global_conf::mv_version_list) * total_tuple_capacity;

    size_t ver_data_mem_req =
        (rec_size * total_ver_capacity) +
        (sizeof(global_conf::mv_version) * total_ver_capacity);

    int list_numa_id = global_conf::delta_list_numa_id;
    int data_numa_id = global_conf::delta_ver_numa_id;
    void* mem_list = MemoryManager::alloc(ver_list_mem_req, list_numa_id);
    void* mem_data = MemoryManager::alloc(ver_data_mem_req, data_numa_id);

    assert(mem_list != nullptr);
    assert(mem_data != nullptr);

    this->ver_list_cursor = (char*)mem_list;
    this->ver_data_cursor = (char*)mem_data;

    // warm-up mem-list
    int* pt = (int*)mem_list;
    int warmup_size = ver_list_mem_req / sizeof(int);
    for (int i = 0; i < warmup_size; i++) pt[i] = 0;

    // warm-up mem-data
    pt = (int*)mem_data;
    warmup_size = ver_data_mem_req / sizeof(int);
    for (int i = 0; i < warmup_size; i++) pt[i] = 0;

    std::cout << "\tDelta size: "
              << ((double)(ver_list_mem_req + ver_data_mem_req) /
                  (1024 * 1024 * 1024))
              << " GB" << std::endl;

    ver_list_data = mem_chunk(mem_list, ver_list_mem_req, list_numa_id);

    ver_data_ptr = mem_chunk(mem_data, ver_data_mem_req, data_numa_id);

    this->rec_size = rec_size;

    used_ver_capacity = 0;
    used_tuple_capacity = 0;

    // std::cout << "Rec size: " << rec_size << std::endl;

    // reserve hash-capacity before hand
    vid_version_map.reserve(total_tuple_capacity);
  }

  // TODO: clear out all the memory
  ~DeltaStore() {
    MemoryManager::free(ver_list_data.data, ver_list_data.size);
    MemoryManager::free(ver_data_ptr.data, ver_data_ptr.size);
  }

  /*void insert_version(uint64_t vid, void* rec, uint64_t tmin, uint64_t tmax) {
    std::unique_lock<std::mutex> lock(this->glbl_lck);
    assert(used_recs_capacity < total_rec_capacity);
    used_recs_capacity++;
    global_conf::mv_version* val = (global_conf::mv_version*)getVersionChunk();
    val->t_min = tmin;
    val->t_max = tmax;
    val->data = getDataChunk();
    memcpy(val->data, rec, rec_size);

    // template <typename K> bool find(const K &key, mapped_type &val)
    {
      std::unique_lock<std::mutex> lock(this->hash_lck);
      global_conf::mv_version_list* vlst;
      vid_version_map.find(vid, vlst);
      vlst->insert(val);
      vid_version_map.insert(vid, vlst);
    }
  }*/

  void* insert_version(uint64_t vid, uint64_t tmin, uint64_t tmax) {
    void* cnk = getVersionDataChunk();
    global_conf::mv_version* val = (global_conf::mv_version*)cnk;
    val->t_min = tmin;
    val->t_max = tmax;
    val->data = (char*)cnk + sizeof(global_conf::mv_version);

    global_conf::mv_version_list* vlst = nullptr;

    if (vid_version_map.find(vid, vlst))
      vlst->insert(val);
    else {
      vlst = (global_conf::mv_version_list*)getListChunk();
      vlst->insert(val);
      vid_version_map.insert(vid, vlst);
    }

    return val->data;
  }

  bool getVersionList(uint64_t vid, global_conf::mv_version_list*& vlst) {
    if (vid_version_map.find(vid, vlst))
      return true;
    else
      assert(false);
    return false;
  }

  global_conf::mv_version_list* getVersionList(uint64_t vid) {
    global_conf::mv_version_list* vlst = nullptr;
    vid_version_map.find(vid, vlst);
    assert(vlst != nullptr);
    return vlst;
  }

  double getUtilPercentage() {
    // TODO: implement

    // return ((double)used_recs_capacity.load() / (double)total_rec_capacity) *
    //       100;

    return 0.0;
  };

  void reset() {
    // THIS IS DANGEROUS AREA BECAUSE THIS ASSUMES THAT NO THREAD IS
    // PERFORMING ANY INSERTS WHEN THIS RESET HAPPENS.

    // std::unique_lock<std::mutex> lock1(this->data_lock, std::defer_lock);
    // std::unique_lock<std::mutex> lock2(this->list_lock, std::defer_lock);
    // std::lock(lock1, lock2);
    vid_version_map.clear();
    ver_list_cursor = (char*)ver_list_data.data;
    ver_data_cursor = (char*)ver_data_ptr.data;
    used_ver_capacity = 0;
    used_tuple_capacity = 0;
  }

 private:
  void* getVersionDataChunk() {
    void* tmp = nullptr;
    {
      std::lock_guard<std::mutex> lock(this->data_lock);
      tmp = (void*)ver_data_cursor;
      ver_data_cursor += rec_size + sizeof(global_conf::mv_version);
    }
    assert(tmp != nullptr);
    return tmp;
  }
  void* getListChunk() {
    void* tmp = nullptr;
    {
      std::lock_guard<std::mutex> lock(this->list_lock);
      tmp = (void*)ver_list_cursor;
      ver_list_cursor += sizeof(global_conf::mv_version_list);
    }
    assert(tmp != nullptr);
    return tmp;
  }

  size_t rec_size;
  uint64_t total_ver_capacity;
  uint64_t total_tuple_capacity;
  std::atomic<uint64_t> used_ver_capacity;
  std::atomic<uint64_t> used_tuple_capacity;

  indexes::HashIndex<uint64_t, global_conf::mv_version_list*> vid_version_map;

  /* Memory Manger*/
  // Currently the delta storage is not extendable so mem_chunk is a object
  // instead of vector.

  char* ver_list_cursor;
  mem_chunk ver_list_data;

  char* ver_data_cursor;
  mem_chunk ver_data_ptr;

  // making them static as we know inserts will be only on the active version
  // plus I was getting more throughput on static. maybe due to bad alignment
  static std::mutex list_lock;
  static std::mutex data_lock;
};

};  // namespace storage

#endif /* DELTA_STORAGE_HPP_ */
