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

#include "operators/mem-move-scaleout.hpp"
// #include "common/gpu/gpu-common.hpp"
// #include "cuda.h"
// #include "cuda_runtime_api.h"
#include <memory/block-manager.hpp>
#include <memory/memory-manager.hpp>

#include "network/infiniband/infiniband-manager.hpp"
#include "threadpool/threadpool.hpp"
#include "util/logging.hpp"

// void MemMoveScaleOut::open(Pipeline *pip) {
//   workunit *wu =
//       (workunit *)MemoryManager::mallocPinned(sizeof(workunit) * slack);

//   // nvtxRangePushA("memmove::open");
//   cudaStream_t strm = createNonBlockingStream();

//   // cudaStream_t strm2;
//   // gpu_run(cudaStreamCreateWithFlags(&strm2, cudaStreamNonBlocking));

//   eventlogger.log(this, log_op::MEMMOVE_OPEN_START);
//   size_t data_size = (pip->getSizeOf(data_type) + 16 - 1) & ~((size_t)0xF);

//   void *pmmc = MemoryManager::mallocPinned(sizeof(MemMoveConf));
//   MemMoveConf *mmc = new (pmmc) MemMoveConf;

//   eventlogger.log(this, log_op::MEMMOVE_OPEN_END);
// #ifndef NCUDA
//   mmc->strm = strm;
//   // mmc->strm2          = strm2;
// #endif
//   mmc->slack = slack;
//   mmc->next_e = 0;
//   // mmc->events         = new cudaEvent_t[slack];
//   // mmc->old_buffs      = new void      *[slack];
//   mmc->data_buffs = MemoryManager::mallocPinned(data_size * slack);
//   char *data_buff = (char *)mmc->data_buffs;
//   eventlogger.log(this, log_op::MEMMOVE_OPEN_START);
//   for (size_t i = 0; i < slack; ++i) {
//     wu[i].data = ((void *)(data_buff + i * data_size));
//     // // gpu_run(cudaEventCreateWithFlags(&(wu[i].event),
//     // cudaEventDisableTiming));//  | cudaEventBlockingSync));
//     //         gpu_run(cudaEventCreate(&(wu[i].event)));
//     //         gpu_run(cudaStreamCreate(&(wu[i].strm)));

//     mmc->idle.push(wu + i);

//     // gpu_run(cudaEventCreateWithFlags(mmc->events + i,
//     // cudaEventDisableTiming
//     // | cudaEventBlockingSync)); gpu_run(cudaEventCreate(mmc->events + i));
//     // mmc->old_buffs[i] = nullptr;
//   }
//   eventlogger.log(this, log_op::MEMMOVE_OPEN_END);
//   // nvtxRangePushA("memmove::open2");
//   for (size_t i = 0; i < slack; ++i) {
//     gpu_run(cudaEventCreateWithFlags(
//         &(wu[i].event), cudaEventDisableTiming | cudaEventBlockingSync));
//     // // gpu_run(cudaEventCreateWithFlags(&(wu[i].event),
//     // cudaEventDisableTiming));//  | cudaEventBlockingSync));
//     //         gpu_run(cudaEventCreate(&(wu[i].event)));
//     //         gpu_run(cudaStreamCreate(&(wu[i].strm)));

//     // gpu_run(cudaEventCreateWithFlags(mmc->events + i,
//     // cudaEventDisableTiming
//     // | cudaEventBlockingSync)); gpu_run(cudaEventCreate(mmc->events + i));
//     // mmc->old_buffs[i] = nullptr;
//   }
//   // nvtxRangePop();

//   eventlogger.log(this, log_op::MEMMOVE_OPEN_START);
//   mmc->worker = ThreadPool::getInstance().enqueue(
//       &MemMoveScaleOut::catcher, this, mmc, pip->getGroup(),
//       exec_location{});
//   // mmc->worker = new thread(&MemMoveDevice::catcher, this, mmc,
//   // pip->getGroup(), exec_location{});
//   eventlogger.log(this, log_op::MEMMOVE_OPEN_END);

//   int device = -1;
//   if (!to_cpu) device = topology::getInstance().getActiveGpu().id;
//   pip->setStateVar<int>(device_id_var, device);

//   // pip->setStateVar<cudaStream_t>(cu_stream_var, strm  );
//   pip->setStateVar<void *>(memmvconf_var, mmc);
//   // nvtxRangePop();
// }

// void MemMoveScaleOut::close(Pipeline *pip) {
//   eventlogger.log(this, log_op::MEMMOVE_CLOSE_START);
//   // int device = get_device();
//   // cudaStream_t strm = pip->getStateVar<cudaStream_t>(cu_stream_var);
//   MemMoveConf *mmc = pip->getStateVar<MemMoveConf *>(memmvconf_var);

//   mmc->tran.close();

//   nvtxRangePop();
//   mmc->worker.get();
//   // mmc->worker->join();

//   eventlogger.log(this, log_op::MEMMOVE_CLOSE_END);

//   eventlogger.log(this, log_op::MEMMOVE_CLOSE_CLEAN_UP_START);
//   // gpu_run(cudaStreamSynchronize(g_strm));

//   // int32_t h_s;
//   // gpu_run(cudaMemcpy(&h_s, s, sizeof(int32_t), cudaMemcpyDefault));
//   // std::cout << "rrr" << h_s << std::endl;

//   // MemoryManager::freeGpu(s);
//   syncAndDestroyStream(mmc->strm);
//   // gpu_run(cudaStreamSynchronize(mmc->strm2));
//   // gpu_run(cudaStreamDestroy    (mmc->strm2));

//   nvtxRangePushA("MemMoveDev_running2");
//   nvtxRangePushA("MemMoveDev_running");

//   nvtxRangePushA("MemMoveDev_release");
//   workunit *start_wu;
//   // void     * start_wu_data;
//   for (size_t i = 0; i < slack; ++i) {
//     workunit *wu = mmc->idle.pop_unsafe();

//     // if (mmc->old_buffs[i])
//     buffer-manager<int32_t>::release_buffer((int32_t
//     // *) mmc->old_buffs[i]);

//     gpu_run(cudaEventDestroy(wu->event));
//     // gpu_run(cudaEventDestroy(mmc->events[i]));
//     // free(wu->data);

//     if (i == 0 || wu < start_wu) start_wu = wu;
//     // if (i == 0 || wu->data < start_wu_data) start_wu_data = wu->data;
//   }
//   nvtxRangePop();
//   nvtxRangePop();

//   MemoryManager::freePinned(mmc->data_buffs);
//   // assert(mmc->tran.empty_unsafe());
//   // assert(mmc->idle.empty_unsafe());
//   // free(start_wu_data);
//   // delete[] start_wu;
//   MemoryManager::freePinned(start_wu);
//   // delete[] mmc->events   ;
//   // delete[] mmc->old_buffs;

//   mmc->idle.close();  // false);

//   // delete mmc->worker;
//   // delete mmc;
//   mmc->~MemMoveConf();
//   MemoryManager::freePinned(mmc);
//   eventlogger.log(this, log_op::MEMMOVE_CLOSE_CLEAN_UP_END);
// }

MemMoveScaleOut::MemMoveConf *MemMoveScaleOut::createMoveConf() const {
  void *pmmc = MemoryManager::mallocPinned(sizeof(MemMoveConf));
  return new (pmmc) MemMoveScaleOut::MemMoveConf;
}

void MemMoveScaleOut::MemMoveConf::propagate(MemMoveDevice::workunit *buff,
                                             bool is_noop) {
  if (!is_noop) {
    // null
  }

  tran.push(buff);

  ++cnt;
  if (cnt % (slack / 2) == 0) InfiniBandManager::flush_read();
}

buff_pair MemMoveScaleOut::MemMoveConf::push(void *src, size_t bytes,
                                             int target_device) {
  auto x = InfiniBandManager::read(src, bytes);
  return buff_pair{x, src};
}

void *MemMoveScaleOut::MemMoveConf::pull(void *buff) {
  return ((subscription *)buff)->wait().data;
}

bool MemMoveScaleOut::MemMoveConf::getPropagated(
    MemMoveDevice::workunit **ret) {
  if (!tran.pop(*ret)) return false;
  return true;
}

void MemMoveScaleOut::close(Pipeline *pip) {
  LOG(INFO) << "closing...";
  InfiniBandManager::flush_read();
  MemMoveDevice::close(pip);
}
