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

#include "operators/router-scaleout.hpp"

#include "memory/block-manager.hpp"
#include "network/infiniband/infiniband-manager.hpp"

using namespace llvm;

void *RouterScaleOut::acquireBuffer(int target, bool polling) {
  return BlockManager::get_buffer();
}

void RouterScaleOut::releaseBuffer(int target, void *buff) {
  // FIXME: we do not have to send the whole block. Usually we will have to send
  // just a couple of bytes. Use that instead
  LOG(INFO) << "writing" << buff;
  InfiniBandManager::write(buff, BlockManager::block_size);
  InfiniBandManager::flush();
}

void RouterScaleOut::freeBuffer(int target, void *buff) {
  // nvtxRangePushA("waiting_to_release");
  // std::unique_lock<std::mutex> lock(free_pool_mutex[target]);
  // nvtxRangePop();
  // free_pool[target].emplace(buff);
  // free_pool_cv[target].notify_one();
  // lock.unlock();
  BlockManager::release_buffer(buff);
}

bool RouterScaleOut::get_ready(int target, void *&buff) {
  LOG(INFO) << "waiting";
  auto &sub = InfiniBandManager::subscribe();
  auto x = sub.wait();
  if (x.size == 0) BlockManager::release_buffer(x.data);
  buff = x.data;
  LOG(INFO) << "received closed " << bytes{x.size};
  return x.size != 0;
}

// void RouterScaleOut::fire(int target, PipelineGen *pipGen) {
//   nvtxRangePushA((pipGen->getName() + ":" + std::to_string(target)).c_str());

//   eventlogger.log(this, log_op::EXCHANGE_CONSUME_OPEN_START);

//   // size_t packets = 0;
//   // time_block t("Xchange pipeline (target=" + std::to_string(target) +
//   "):");

//   eventlogger.log(this, log_op::EXCHANGE_CONSUME_OPEN_END);
//   set_exec_location_on_scope d(target_processors[target]);
//   Pipeline *pip = pipGen->getPipeline(target);
//   std::this_thread::yield();  // if we remove that, following opens may
//   allocate
//   // memory to wrong socket!

//   nvtxRangePushA(
//       (pipGen->getName() + ":" + std::to_string(target) + "open").c_str());
//   pip->open();
//   nvtxRangePop();
//   eventlogger.log(this, log_op::EXCHANGE_CONSUME_OPEN_START);

//   eventlogger.log(this, log_op::EXCHANGE_CONSUME_OPEN_END);

//   {
//     do {
//       void *p;
//       if (!get_ready(target, p)) break;
//       nvtxRangePushA((pipGen->getName() + ":cons").c_str());

//       pip->consume(0, p);
//       nvtxRangePop();

//       freeBuffer(target, p);  // FIXME: move this inside the generated code

//       // std::this_thread::yield();
//     } while (true);
//   }

//   eventlogger.log(this, log_op::EXCHANGE_CONSUME_CLOSE_START);

//   nvtxRangePushA(
//       (pipGen->getName() + ":" + std::to_string(target) + "close").c_str());
//   pip->close();
//   nvtxRangePop();

//   // std::cout << "Xchange pipeline packets (target=" << target << "): " <<
//   // packets << std::endl;

//   nvtxRangePop();

//   eventlogger.log(this, log_op::EXCHANGE_CONSUME_CLOSE_END);
// }

// void RouterScaleOut::open(Pipeline *pip) {
//   // time_block t("Tinit_exchange: ");

//   std::lock_guard<std::mutex> guard(init_mutex);

//   if (firers.empty()) {
//     for (int i = 0; i < numOfParents; ++i) {
//       ready_fifo[i].reset();
//     }

//     eventlogger.log(this, log_op::EXCHANGE_INIT_CONS_START);
//     remaining_producers = producers;
//     for (int i = 0; i < numOfParents; ++i) {
//       firers.emplace_back(&RouterScaleOut::fire, this, i, catch_pip);
//     }
//     eventlogger.log(this, log_op::EXCHANGE_INIT_CONS_END);
//   }
// }

void RouterScaleOut::fire_close(Pipeline *pip) {
  // time_block t("Tterm_exchange: ");
  LOG(INFO) << "closing";
}

void RouterScaleOut::close(Pipeline *pip) {
  // time_block t("Tterm_exchange: ");
  LOG(INFO) << "close";
  // Send msg: "You are now allowed to proceed to your close statement"
  InfiniBandManager::write(BlockManager::get_buffer(), 0);
  InfiniBandManager::flush();

  int rem = --remaining_producers;
  assert(rem >= 0);

  if (rem == 0) {
    eventlogger.log(this, log_op::EXCHANGE_JOIN_START);
    nvtxRangePushA("Exchange_waiting_to_close");
    for (auto &t : firers) t.get();
    nvtxRangePop();
    eventlogger.log(this, log_op::EXCHANGE_JOIN_END);
    firers.clear();
  }

  // Send msg: I am done and I will exit as soon as you all tell me you are also
  // done
  InfiniBandManager::write(BlockManager::get_buffer(), 2);
  InfiniBandManager::flush();

  LOG(INFO) << "waiting for other end...";
  auto &sub = InfiniBandManager::subscribe();
  auto x = sub.wait();
  LOG(INFO) << "received closed " << x.size;
}
