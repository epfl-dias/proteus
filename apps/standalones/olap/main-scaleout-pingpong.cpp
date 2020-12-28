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

#include <arpa/inet.h>
#include <gflags/gflags.h>
#include <rdma/rdma_cma.h>
#include <sys/types.h>

#include <olap/plan/prepared-statement.hpp>
#include <platform/common/error-handling.hpp>
#include <platform/memory/block-manager.hpp>
#include <platform/memory/memory-manager.hpp>
#include <platform/network/infiniband/infiniband-manager.hpp>
#include <platform/storage/storage-manager.hpp>
#include <platform/topology/affinity_manager.hpp>
#include <platform/topology/topology.hpp>
#include <platform/util/logging.hpp>
#include <platform/util/profiling.hpp>
#include <platform/util/timing.hpp>
#include <type_traits>

#include "cli-flags.hpp"

void handshake(subscription &sub) {
  for (int i = 0; i < 1; ++i) {
    if (FLAGS_primary) {
      LOG(INFO) << "send";
      auto ptr = BlockManager::get_buffer();
      ((int *)ptr.get())[0] = 45;
      InfiniBandManager::send(std::move(ptr), 4);
      LOG(INFO) << "send done";
    } else {
      sleep(2);
      LOG(INFO) << "wait";
      sub.wait();
      LOG(INFO) << "wait done";
      //      auto v = sub.wait();
      //      BlockManager::release_buffer((int32_t *) v.data);
    }

    if (FLAGS_primary) {
      sub.wait();
      //      auto v = sub.wait();
      //      BlockManager::release_buffer((int32_t *) v.data);
    } else {
      LOG(INFO) << "send";
      auto ptr = BlockManager::get_buffer();
      ((int *)ptr.get())[0] = 44;
      InfiniBandManager::send(std::move(ptr), 4);
      LOG(INFO) << "send done";
    }
  }
}

template <typename T>
void repeat_q(T eval) {
  for (size_t k = 0; k < FLAGS_repeat; ++k) {
    eval();
  }
}

template <typename T>
void iterate_over_buffsizes(T eval) {
  for (size_t buff_size = BlockManager::block_size; buff_size >= 4 * 1024;
       buff_size /= 2) {
    LOG(INFO) << "Buffer size: " << bytes{buff_size};
    eval(buff_size);
  }
}

int main(int argc, char *argv[]) {
  auto ctx = proteus::from_cli::olap(
      "Simple command line interface for proteus", &argc, &argv);

  // set_exec_location_on_scope affg{topology::getInstance().getGpus()[1]};
  set_exec_location_on_scope aff{
      topology::getInstance().getCpuNumaNodes()[FLAGS_primary ? 0 : 1]};

  assert(FLAGS_port <= std::numeric_limits<uint16_t>::max());
  InfiniBandManager::init(FLAGS_url, static_cast<uint16_t>(FLAGS_port),
                          FLAGS_primary, FLAGS_ipv4);

  auto &sub = InfiniBandManager::subscribe();

  handshake(sub);

  if (FLAGS_primary) {
    void *v_data;
    auto v_size = (size_t{25} * 1024) * 1024 * 1024;
    {
      // v_data = mmap(nullptr, v_size, PROT_READ | PROT_WRITE,
      //               MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
      // assert(v_data != MAP_FAILED);
      v_data = MemoryManager::mallocPinned(v_size);
      auto x = InfiniBandManager::reg(v_data, v_size);
      LOG(INFO) << x.first << " " << x.second;

      auto ptr = BlockManager::get_buffer();
      ((decltype(x) *)ptr.get())[0] = x;
      InfiniBandManager::send(std::move(ptr), sizeof(decltype(x)));
    }

    sub.wait();
    iterate_over_buffsizes([&](auto buff_size) {
      repeat_q([&] {
        time_block t{"Tg:w:" + std::to_string(bytes{buff_size}) + ": "};
        int32_t sum = 0;

        {
          time_block t{"T-:"};
          // time_block t{"Tg-nosend:" + std::to_string(bytes{buff_size}) +
          //              ": "};
          nvtxRangePushA("reading");
          do {
            eventlogger.log(nullptr, IB_WAITING_DATA_START);
            nvtxRangePushA("waiting");
            auto x = sub.wait();
            nvtxRangePop();
            eventlogger.log(nullptr, IB_WAITING_DATA_END);
            if (x.size == 0) break;
            assert(x.size % 4 == 0);
            size_t size = x.size / 4;
            auto *data = (int32_t *)x.data.get();
            if (data) {
              for (size_t i = 0; i < size; ++i) {
                sum += data[i];
              }
            }
            // MemoryManager::freePinned(sub.wait().data);
            BlockManager::release_buffer(std::move(x.data));
          } while (true);
        }
        nvtxRangePop();
        // std::cout << sum << std::endl;
        auto f = BlockManager::get_buffer();
        ((int32_t *)f.get())[0] = sum;
        InfiniBandManager::send(std::move(f), 4);
        // std::cout << sum << std::endl;
      });
      // for (size_t k = 0; k < 1; ++k) {
      //   time_block t{"Tg:r:" + std::to_string(bytes{buff_size}) + ": "};
      //   int32_t sum = 0;
      //   auto x = sub.wait();
      //   auto v = ((std::pair<int32_t *, size_t> *)x.data)[0];
      //   auto data = v.first;
      //   auto size = v.second;

      //   {
      //     std::deque<typename std::result_of<decltype (
      //         &InfiniBandManager::read)(void *, size_t)>::type>
      //         futures;
      //     size_t slack = 1024;
      //     // {
      //     //   time_block t{"T: "};

      //     //   for (size_t i = 0; i < size; i += buff_size) {
      //     //     futures.emplace_back(InfiniBandManager::read(
      //     //         ((char *)data) + i, std::min(buff_size, size - i)));
      //     //   }
      //     //   InfiniBandManager::flush_read();
      //     // }
      //     // for (auto &t : futures) {
      //     //   BlockManager::release_buffer(t->wait().data);
      //     // }
      //     for (size_t i = 0; i < std::min(slack * buff_size, size);
      //          i += buff_size) {
      //       futures.emplace_back(InfiniBandManager::read(
      //           ((char *)data) + i, std::min(buff_size, size - i)));
      //     }
      //     bool flushed = slack * buff_size >= size;
      //     if (flushed) {
      //       InfiniBandManager::flush_read();
      //     }
      //     for (size_t i = 0; i < size; i += buff_size) {
      //       size_t i_next = i + buff_size * slack;
      //       if (i_next < size) {
      //         futures.emplace_back(InfiniBandManager::read(
      //             ((char *)data) + i_next, std::min(buff_size, size -
      //             i_next)));
      //       } else {
      //         if (!flushed) InfiniBandManager::flush_read();
      //         flushed = true;
      //       }
      //       assert(!futures.empty());
      //       // auto x = futures.front().get();
      //       auto x = futures.front()->wait().data;
      //       futures.pop_front();
      //       {
      //         int32_t *data = (int32_t *)x;
      //         size_t s = std::min(size_t{1}, size - i) / 4;
      //         for (size_t i = 0; i < s; ++i) {
      //           sum += data[i];
      //         }
      //         BlockManager::release_buffer(data);
      //       }
      //     }
      //   }
      //   InfiniBandManager::flush();

      //   // std::cout << sum << std::endl;
      //   auto f = BlockManager::get_buffer();
      //   f[0] = sum;
      //   InfiniBandManager::send(f, 4);
      //   // std::cout << sum << std::endl;
      // }
    });
    InfiniBandManager::unreg(v_data);
    // munmap(v_data, v_size);
    MemoryManager::freePinned(v_data);
  } else {
    void *v_data;
    auto v_size = (size_t{4} * 1024) * 1024 * 1024;
    {
      auto x = sub.wait();
      assert(x.size == sizeof(buffkey));
      auto kb = ((buffkey *)x.data.release())[0];
      LOG(INFO) << kb.first << " " << kb.second;

      auto v = StorageManager::getInstance()
                   .getOrLoadFile("inputs/ssbm1000/lineorder.csv.lo_orderdate",
                                  4, PINNED)
                   .get();

      {
        //        v_data = mmap(nullptr, v_size, PROT_READ | PROT_WRITE,
        //                      MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1,
        //                      0);
        //        assert(v_data != MAP_FAILED);
        //        // v_data = MemoryManager::mallocPinned(v_size);
        //        CUdeviceptr d_A{};
        //        gpu_run(cuMemAlloc(&d_A, v_size));
        v_data = MemoryManager::mallocGpu(v_size);  // v[0].size);
        //        v_data = (void *) d_A;
        assert(v_size >= v[0].size);
        InfiniBandManager::reg(v_data, v_size);
        gpu_run(cudaMemcpy(v_data, v[0].data, v[0].size, cudaMemcpyDefault));
        //        memcpy(v_data, v[0].data, v[0].size);
      }
      LOG(INFO) << v_data << " " << bytes{v_size};
      // InfiniBandManager::reg((void *)v[0].data, v[0].size);
      profiling::resume();
      InfiniBandManager::send(proteus::managed_ptr{v_data}, 0);
      // int32_t j = 0;
      assert(v.size() == 1);
      iterate_over_buffsizes([&](auto buff_size) {
        repeat_q([&] {
          time_block t{"Tg:" + std::to_string(bytes{buff_size}) + ": "};

          {
            time_block t{"T: "};
            for (size_t i = 0; i < v[0].size; i += buff_size) {
              InfiniBandManager::write_to(
                  proteus::managed_ptr{((char *)v_data) + i},
                  std::min(buff_size, v[0].size - i),
                  {((char *)kb.first) + i, kb.second});
              // InfiniBandManager::write(((char *)v_data) + i,
              //                          std::min(buff_size, v[0].size - i));
            }
          }
          InfiniBandManager::write(proteus::managed_ptr{v_data}, 0);
          InfiniBandManager::flush();

          auto x = sub.wait();
          LOG(INFO) << ((int32_t *)x.data.release())[0] << std::endl;
        });
        // for (size_t k = 0; k < 1; ++k) {
        //   time_block t{"Tg:read:" + std::to_string(bytes{buff_size}) + ": "};
        //   auto f = BlockManager::get_buffer();
        //   auto p = std::make_pair(v[0].data, v[0].size);
        //   ((decltype(&p))f)[0] = p;
        //   InfiniBandManager::write(f, sizeof(p));
        //   InfiniBandManager::flush();

        //   auto x = sub.wait();
        //   std::cout << ((int32_t *)x.data)[0] << std::endl;
        //   BlockManager::release_buffer((int32_t *)x.data);
        // }
      });
    }
    InfiniBandManager::unreg(v_data);
    MemoryManager::freeGpu(v_data);
    //    munmap(v_data, v_size);
    profiling::pause();

    std::cout << "DONE" << std::endl;
    InfiniBandManager::disconnectAll();
    StorageManager::getInstance().unloadAll();
  }

  StorageManager::getInstance().unloadAll();
  InfiniBandManager::deinit();
  return 0;

  // if (FLAGS_primary) {
  //   auto x = sub.wait();
  //   BlockManager::release_buffer((int32_t *)x.data);
  //   for (size_t buff_size = BlockManager::block_size; buff_size >= 4 * 1024;
  //        buff_size /= 2) {
  //     LOG(INFO) << "Packet size: " << bytes{buff_size};
  //     for (size_t k = 0; k < 1; ++k) {
  //       time_block t{"Tg:w:" + std::to_string(bytes{buff_size}) + ": "};
  //       int32_t sum = 0;

  //       {
  //         // time_block t{"Tg-nosend:" + std::to_string(bytes{buff_size}) +
  //         //              ": "};
  //         nvtxRangePushA("reading");
  //         do {
  //           eventlogger.log(nullptr, IB_WAITING_DATA_START);
  //           nvtxRangePushA("waiting");
  //           auto x = sub.wait();
  //           nvtxRangePop();
  //           eventlogger.log(nullptr, IB_WAITING_DATA_END);
  //           if (x.size == 0) break;
  //           assert(x.size % 4 == 0);
  //           size_t size = x.size / 4;
  //           int32_t *data = (int32_t *)x.data;
  //           for (size_t i = 0; i < size; ++i) {
  //             sum += data[i];
  //           }
  //           // MemoryManager::freePinned(sub.wait().data);
  //           BlockManager::release_buffer(data);
  //         } while (true);
  //       }
  //       nvtxRangePop();
  //       // std::cout << sum << std::endl;
  //       auto f = BlockManager::get_buffer();
  //       f[0] = sum;
  //       InfiniBandManager::send(f, 4);
  //       // std::cout << sum << std::endl;
  //     }
  //     for (size_t k = 0; k < 1; ++k) {
  //       time_block t{"Tg:r:" + std::to_string(bytes{buff_size}) + ": "};
  //       int32_t sum = 0;
  //       auto x = sub.wait();
  //       auto v = ((std::pair<int32_t *, size_t> *)x.data)[0];
  //       auto data = v.first;
  //       auto size = v.second;

  //       {
  //         std::deque<typename std::result_of<decltype (
  //             &InfiniBandManager::read)(void *, size_t)>::type>
  //             futures;
  //         size_t slack = 1024;
  //         // {
  //         //   time_block t{"T: "};

  //         //   for (size_t i = 0; i < size; i += buff_size) {
  //         //     futures.emplace_back(InfiniBandManager::read(
  //         //         ((char *)data) + i, std::min(buff_size, size - i)));
  //         //   }
  //         //   InfiniBandManager::flush_read();
  //         // }
  //         // for (auto &t : futures) {
  //         //   BlockManager::release_buffer(t->wait().data);
  //         // }
  //         for (size_t i = 0; i < std::min(slack * buff_size, size);
  //              i += buff_size) {
  //           futures.emplace_back(InfiniBandManager::read(
  //               ((char *)data) + i, std::min(buff_size, size - i)));
  //         }
  //         bool flushed = slack * buff_size >= size;
  //         if (flushed) {
  //           InfiniBandManager::flush_read();
  //         }
  //         for (size_t i = 0; i < size; i += buff_size) {
  //           size_t i_next = i + buff_size * slack;
  //           if (i_next < size) {
  //             futures.emplace_back(InfiniBandManager::read(
  //                 ((char *)data) + i_next, std::min(buff_size, size -
  //                 i_next)));
  //           } else {
  //             if (!flushed) InfiniBandManager::flush_read();
  //             flushed = true;
  //           }
  //           assert(!futures.empty());
  //           // auto x = futures.front().get();
  //           auto x = futures.front()->wait().data;
  //           futures.pop_front();
  //           {
  //             int32_t *data = (int32_t *)x;
  //             size_t s = std::min(size_t{1}, size - i) / 4;
  //             for (size_t i = 0; i < s; ++i) {
  //               sum += data[i];
  //             }
  //             BlockManager::release_buffer(data);
  //           }
  //         }
  //       }
  //       InfiniBandManager::flush();

  //       // std::cout << sum << std::endl;
  //       auto f = BlockManager::get_buffer();
  //       f[0] = sum;
  //       InfiniBandManager::send(f, 4);
  //       // std::cout << sum << std::endl;
  //     }
  //   }
  // } else {
  //   auto v = StorageManager::getOrLoadFile(
  //       "inputs/ssbm100/lineorder.csv.lo_orderdate", 4, PINNED);

  //   // InfiniBandManager::reg((void *)v[0].data, v[0].size);
  //   profiling::resume();
  //   InfiniBandManager::send((char *)v[0].data, 0);
  //   // int32_t j = 0;
  //   assert(v.size() == 1);
  //   for (size_t buff_size = BlockManager::block_size; buff_size >= 4 * 1024;
  //        buff_size /= 2) {
  //     LOG(INFO) << "Packet size: " << bytes{buff_size};
  //     for (size_t k = 0; k < 1; ++k) {
  //       time_block t{"Tg:" + std::to_string(bytes{buff_size}) + ": "};

  //       {
  //         time_block t{"T: "};
  //         for (size_t i = 0; i < v[0].size; i += buff_size) {
  //           InfiniBandManager::write(((char *)v[0].data) + i,
  //                                    std::min(buff_size, v[0].size - i));
  //         }
  //       }
  //       InfiniBandManager::write((char *)v[0].data, 0);
  //       InfiniBandManager::flush();

  //       auto x = sub.wait();
  //       std::cout << ((int32_t *)x.data)[0] << std::endl;
  //       BlockManager::release_buffer((int32_t *)x.data);
  //     }
  //     for (size_t k = 0; k < 1; ++k) {
  //       time_block t{"Tg:read:" + std::to_string(bytes{buff_size}) + ": "};
  //       auto f = BlockManager::get_buffer();
  //       auto p = std::make_pair(v[0].data, v[0].size);
  //       ((decltype(&p))f)[0] = p;
  //       InfiniBandManager::write(f, sizeof(p));
  //       InfiniBandManager::flush();

  //       auto x = sub.wait();
  //       std::cout << ((int32_t *)x.data)[0] << std::endl;
  //       BlockManager::release_buffer((int32_t *)x.data);
  //     }
  //   }
  //   profiling::pause();

  //   std::cout << "DONE" << std::endl;
  //   InfiniBandManager::disconnectAll();
  //   InfiniBandManager::unreg((void *)v[0].data);
  //   StorageManager::unloadAll();
  // }

  // InfiniBandManager::deinit();
  // StorageManager::unloadAll();
  // MemoryManager::destroy();
  // return 0;
}
