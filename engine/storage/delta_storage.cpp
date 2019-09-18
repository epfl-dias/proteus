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

#include "storage/delta_storage.hpp"

#include <sys/mman.h>

#include "scheduler/worker.hpp"

namespace storage {

/*
std::chrono::time_point<std::chrono::system_clock,
                              std::chrono::nanoseconds>
          start_time;

      vid_version_map.clear();

      std::chrono::duration<double> diff =
          std::chrono::system_clock::now() - start_time;

      std::cout << "version clear time: " << diff.count() << std::endl;

*/

DeltaStore::DeltaStore(uint delta_id, uint64_t ver_list_capacity,
                       uint64_t ver_data_capacity, int num_partitions)
    : touched(false) {
  this->delta_id = delta_id;

  ver_list_capacity = ver_list_capacity * (1024 * 1024 * 1024);  // GB
  ver_list_capacity = ver_list_capacity / 2;
  ver_data_capacity = ver_data_capacity * (1024 * 1024 * 1024);  // GB
  for (int i = 0; i < num_partitions; i++) {
    uint list_numa_id = 0;  // i % NUM_SOCKETS;
    uint data_numa_id = 0;  // i % NUM_SOCKETS;

    // std::cout << "PID-" << i << " - memset: " << data_numa_id << std::endl;

    void* mem_list = MemoryManager::alloc(ver_list_capacity, i,
                                          MADV_DONTFORK | MADV_HUGEPAGE);
    void* mem_data = MemoryManager::alloc(ver_data_capacity, i,
                                          MADV_DONTFORK | MADV_HUGEPAGE);
    assert(mem_list != NULL);
    assert(mem_data != NULL);

    // assert(mlock(mem_list, ver_list_mem_req));
    // assert(mlock(mem_data, ver_data_mem_req));

    assert(mem_list != nullptr);
    assert(mem_data != nullptr);

    void* obj_data = MemoryManager::alloc(sizeof(DeltaPartition), i);

    partitions.emplace_back(new (obj_data) DeltaPartition(
        (char*)mem_list, mem_chunk(mem_list, ver_list_capacity, i),
        (char*)mem_data, mem_chunk(mem_data, ver_data_capacity, i), i));
  }

  if (DELTA_DEBUG) {
    std::cout << "\tDelta size: "
              << ((double)(ver_list_capacity + ver_data_capacity) /
                  (1024 * 1024 * 1024))
              << " GB * " << num_partitions << " Partitions" << std::endl;
    std::cout << "\tDelta size: "
              << ((double)(ver_list_capacity + ver_data_capacity) *
                  num_partitions / (1024 * 1024 * 1024))
              << " GB" << std::endl;
  }
  this->total_mem_reserved =
      (ver_list_capacity + ver_data_capacity) * num_partitions;

  // for (int i = 0; i < MAX_WORKERS; i++) {
  //   read_ctr[i] = 0;
  // }

  // reserve hash-capacity before hand
  // vid_version_map.reserve(10000000);
  this->readers.store(0);
  this->gc_reset_success.store(0);
  this->gc_requests.store(0);
  this->ops.store(0);
  this->gc_lock.store(0);
  this->tag = 1;
  this->max_active_epoch = 0;
  // this->min_active_epoch = std::numeric_limits<uint64_t>::max();
}

DeltaStore::~DeltaStore() { print_info(); }

void DeltaStore::print_info() {
  static int i = 0;
  std::cout << "[DeltaStore # " << i
            << "] Number of GC Requests: " << this->gc_requests.load()
            << std::endl;

  std::cout << "[DeltaStore # " << i << "] Number of Successful GC Resets: "
            << this->gc_reset_success.load() << std::endl;
  std::cout << "[DeltaStore # " << i
            << "] Number of Operations: " << this->ops.load() << std::endl;

  for (auto& p : partitions) {
    p->report();
  }
  i++;
  if (i >= partitions.size()) i = 0;
}

void* DeltaStore::insert_version(global_conf::IndexVal* idx_ptr, uint rec_size,
                                 ushort parition_id) {
  char* cnk = (char*)partitions[parition_id]->getVersionDataChunk(rec_size);
  global_conf::mv_version* val = (global_conf::mv_version*)cnk;
  val->t_min = idx_ptr->t_min;
  val->t_max = 0;  // idx_ptr->t_max;
  val->data = cnk + sizeof(global_conf::mv_version);

  if (idx_ptr->delta_ver_tag != tag) {
    // none/stale list
    idx_ptr->delta_ver_tag = tag;
    idx_ptr->delta_ver =
        (global_conf::mv_version_list*)partitions[parition_id]->getListChunk();
    idx_ptr->delta_ver->insert(val);
  } else {
    // valid list
    idx_ptr->delta_ver->insert(val);
  }

  if (!touched) touched = true;
  return val->data;
}

// void* DeltaStore::insert_version(uint64_t vid, uint64_t tmin, uint64_t tmax,
//                                  ushort rec_size, ushort parition_id) {
//   // void* cnk = getVersionDataChunk();

//   // while (gc_lock != 0) dont need a gc lock, if someone is here means the
//   // read_counter is already +1 so never gonna gc
//   // std::cout << "--" << parition_id << "--" << std::endl;
//   char* cnk = (char*)partitions[parition_id]->getVersionDataChunk(rec_size);
//   global_conf::mv_version* val = (global_conf::mv_version*)cnk;
//   val->t_min = tmin;
//   val->t_max = tmax;
//   val->data = cnk + sizeof(global_conf::mv_version);

//   // global_conf::mv_version_list* vlst = nullptr;

//   std::pair<int, global_conf::mv_version_list*> v_pair(-1, nullptr);

//   if (vid_version_map.find(vid, v_pair)) {
//     if (v_pair.first == this->tag) {
//       // valid list
//       v_pair.second->insert(val);
//     } else {
//       // invalid list
//       // int tmp = v_pair.first;
//       v_pair.first = tag;
//       v_pair.second = (global_conf::mv_version_list*)partitions[parition_id]
//                           ->getListChunk();
//       v_pair.second->insert(val);
//       vid_version_map.update(vid, v_pair);
//     }

//   } else {
//     // new record overall
//     v_pair.first = tag;
//     v_pair.second =
//         (global_conf::mv_version_list*)partitions[parition_id]->getListChunk();
//     v_pair.second->insert(val);
//     vid_version_map.insert(vid, v_pair);
//   }
//   // ops++;
//   if (!touched) touched = true;
//   return val->data;
// }

// bool DeltaStore::getVersionList(uint64_t vid,
//                                 global_conf::mv_version_list*& vlst) {
//   std::pair<int, global_conf::mv_version_list*> v_pair(-1, nullptr);
//   if (vid_version_map.find(vid, v_pair)) {
//     if (v_pair.first == tag) {
//       vlst = v_pair.second;
//       return true;
//     }
//   }
//   assert(false);

//   return false;

//   // if (vid_version_map.find(vid, vlst))
//   //   return true;
//   // else
//   //   assert(false);
//   // return false;
// }

// global_conf::mv_version_list* DeltaStore::getVersionList(uint64_t vid) {
//   std::pair<int, global_conf::mv_version_list*> v_pair(-1, nullptr);
//   vid_version_map.find(vid, v_pair);

//   // if (v_pair.first != tag) {
//   //   std::cout << "first: " << v_pair.first << std::endl;
//   //   std::cout << "tag: " << tag << std::endl;
//   // }

//   assert(v_pair.first == tag);
//   return v_pair.second;

//   // global_conf::mv_version_list* vlst = nullptr;
//   // vid_version_map.find(vid, vlst);
//   // assert(vlst != nullptr);
//   // return vlst;
// }

// void reset() {
//   vid_version_map.clear();
//   for (auto& p : partitions) {
//     p->reset();
//   }
// }

void DeltaStore::gc() {
  // std::cout << "." << std::endl;
  short e = 0;
  if (gc_lock.compare_exchange_strong(e, -1)) {
    // gc_requests++;

    uint64_t last_alive_txn =
        scheduler::WorkerPool::getInstance().get_min_active_txn();

    // missing condition: or space > 90%
    if (this->readers == 0 && should_gc() &&
        last_alive_txn > max_active_epoch) {
      // std::cout << "delta_id#: " << delta_id << std::endl;
      // std::cout << "request#: " << gc_requests << std::endl;
      // std::cout << "last_alive_txn: " << last_alive_txn << std::endl;
      // std::cout << "max_active_epoch: " << max_active_epoch << std::endl;

      // std::chrono::time_point<std::chrono::system_clock,
      //                         std::chrono::nanoseconds>
      //     start_time;

      // vid_version_map.clear();

      // std::chrono::duration<double> diff =
      //     std::chrono::system_clock::now() - start_time;

      // std::cout << "version clear time: " << diff.count() << std::endl;
      for (auto& p : partitions) {
        p->reset();
      }
      tag++;

      // if (tag % VER_CLEAR_ITER == 0) {
      //   vid_version_map.clear();
      // }
      // gc_lock.unlock();
      gc_lock.store(0);
      touched = false;
      // gc_reset_success++;
    } else {
      // gc_lock.unlock();
      gc_lock.store(0);
    }
  }
}

// void DeltaStore::gc_with_counter_arr(int wrk_id) {
//   // optimization: start with your own socket and then look for readers on
//   // other scoket. second optimization, keep a read counter per partition but
//   // atomic/volatile maybe.

//   short e = 0;
//   gc_requests++;

//   if (gc_lock.compare_exchange_strong(e, -1)) {
//     bool go = true;

//     // for (int i = 0; i < MAX_WORKERS / 8; i += 8) {
//     //   uint64_t* t = (uint64_t*)(read_ctr + (i * 8));
//     //   if (*t != 0) {
//     //     go = false;
//     //     // break;
//     //   }
//     // }
//     //#pragma clang loop vectorize(enable)
//     for (int i = 0; i < MAX_WORKERS; i++) {
//       if (read_ctr[i] != 0) {
//         go = false;
//         // break;
//       }
//     }
//     uint64_t last_alive_txn =
//         scheduler::WorkerPool::getInstance().get_min_active_txn();
//     if (go && last_alive_txn > max_active_epoch) {
//       // vid_version_map.clear();
//       tag += 1;
//       // if (tag % VER_CLEAR_ITER == 0) {
//       //   vid_version_map.clear();
//       // }
//       for (auto& p : partitions) {
//         p->reset();
//       }
//       gc_reset_success++;
//     }
//     gc_lock.store(0);
//   }
// }

}  // namespace storage
