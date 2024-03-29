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

#include "router-scaleout.hpp"

#include <platform/memory/block-manager.hpp>
#include <platform/network/infiniband/infiniband-manager.hpp>
#include <platform/util/timing.hpp>

#include "lib/util/jit/pipeline.hpp"

using namespace llvm;

static std::queue<void *> pending;

proteus::managed_ptr RouterScaleOut::acquireBuffer(int target, bool polling) {
  if (target == InfiniBandManager::server_id()) {
    return Router::acquireBuffer(target, polling);
  } else {
    return proteus::managed_ptr{BlockManager::get_buffer()};
  }
}

void RouterScaleOut::releaseBuffer(int target, proteus::managed_ptr buff) {
  // FIXME: we do not have to send the whole block. Usually we will have to send
  //  just a couple of bytes. Use that instead
  if (target == InfiniBandManager::server_id()) {
    Router::releaseBuffer(target, std::move(buff));
  } else {
    InfiniBandManager::write(std::move(buff), buf_size, sub->id);
    ++cnt;
    if (cnt % (slack / (slack / 16)) == 0) InfiniBandManager::flush();
  }
}

void RouterScaleOut::freeBuffer(int target, proteus::managed_ptr buff) {
  if (!(((uintptr_t)buff.get()) & 1u)) {
    Router::freeBuffer(InfiniBandManager::server_id(), std::move(buff));
    //    Router::freeBuffer(target, buff);
  } else {
    //    LOG(INFO) << buff;
    BlockManager::release_buffer(proteus::managed_ptr{
        (void *)(((uintptr_t)buff.release()) & ~uintptr_t{1})});
    ////    assert(false);
    //    Router::freeBuffer(InfiniBandManager::server_id(), buff);
    ////    Router::freeBuffer(InfiniBandManager::server_id(), buff);
    //    BlockManager::release_buffer(buff);
  }
}

bool RouterScaleOut::get_ready(int target, proteus::managed_ptr &buff) {
  if (target == InfiniBandManager::server_id()) {
    return Router::get_ready(target, buff);
  } else {
    while (true) {
      // auto &sub = InfiniBandManager::subscribe();
      auto x = sub->wait();
      if (x.size == 3) {
        LOG(INFO) << "exit and close local queue if needed " << sub->id << " "
                  << closed;
        if (++closed == 2) {
          ready_fifo[InfiniBandManager::server_id()].close();
          LOG(INFO) << "closed from here";
        }
        strmclosed = true;
        BlockManager::release_buffer(x.release());
        return false;
      }
      assert(x.size != 2);
      Router::releaseBuffer(
          InfiniBandManager::server_id(),
          proteus::managed_ptr{
              (void *)(((uintptr_t)x.release().release()) | 1u)});
    }
  }
}

void RouterScaleOut::fire(int target, PipelineGen *pipGen,
                          const void *session) {
  if (target == InfiniBandManager::server_id()) {
    return Router::fire(target, pipGen, session);
  }

  nvtxRangePushA((pipGen->getName() + ":" + std::to_string(target)).c_str());

  const auto &cu = aff->getAvailableCU(target);
  // set_exec_location_on_scope d(cu);
  auto exec_affinity = cu.set_on_scope();
  std::this_thread::yield();  // if we remove that, following opens may allocate
                              // memory to wrong socket!

  proteus::managed_ptr p;
  while (get_ready(target, p)) {
    freeBuffer(target, std::move(p));
  }
}

void RouterScaleOut::open(Pipeline *pip) {
  // time_block t("Tinit_exchange: ");

  std::lock_guard<std::mutex> guard(init_mutex);

  if (firers.empty()) {
    free_pool = new threadsafe_set<void *>[fanout];
    ready_fifo = new AsyncQueueMPSC<void *>[fanout];
    assert(free_pool);

    for (int i = 0; i < 2; ++i) {
      ready_fifo[i].reset();
    }

    sub = &InfiniBandManager::create_subscription();
    strmclosed = false;

    eventlogger.log(this, log_op::EXCHANGE_INIT_CONS_START);
    remaining_producers = 1;
    for (int i = 0; i < 2; ++i) {
      firers.emplace_back(&RouterScaleOut::fire, this, i, catch_pip,
                          pip->getSession());
    }
    eventlogger.log(this, log_op::EXCHANGE_INIT_CONS_END);
  }
}

void RouterScaleOut::close(Pipeline *pip) {
  // time_block t("Tterm_exchange: ");
  if (++closed == 2) {
    InfiniBandManager::flush();
    ready_fifo[InfiniBandManager::server_id()].close();
  }
  InfiniBandManager::flush();
  // Send msg: "You are now allowed to proceed to your close statement"
  InfiniBandManager::write(proteus::managed_ptr{BlockManager::get_buffer()}, 3,
                           sub->id);
  InfiniBandManager::flush();

  int rem = --remaining_producers;
  assert(rem >= 0);

  if (rem == 0) {
    //    eventlogger.log(this, log_op::EXCHANGE_JOIN_START);
    nvtxRangePushA("Exchange_waiting_to_close");
    for (auto &t : firers) t.get();
    nvtxRangePop();
    //    eventlogger.log(this, log_op::EXCHANGE_JOIN_END);
    firers.clear();
  }

  // Send msg: I am done and I will exit as soon as you all tell me you are also
  // done
  InfiniBandManager::write(proteus::managed_ptr{BlockManager::get_buffer()}, 2,
                           sub->id);
  InfiniBandManager::flush();

  LOG(INFO) << "waiting for other end..." << sub->id;
  // auto &sub = InfiniBandManager::subscribe();
  auto x = sub->wait();
  assert(x.size == 2);
  LOG(INFO) << "received closed " << x.size;
  LOG(INFO) << "data: " << (bytes{buf_size * cnt});
  BlockManager::release_buffer(x.release());
  cnt = 0;
  closed = 0;

  if (rem == 0) {
    delete[] free_pool;
    delete[] ready_fifo;
  }
  sub = nullptr;
}
