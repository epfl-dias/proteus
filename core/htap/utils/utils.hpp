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

#ifndef HARMONIA_UTILS_HPP_
#define HARMONIA_UTILS_HPP_

#include <signal.h>
#include <unistd.h>
#include <iostream>
#include <string>
#include <thread>

#include "cli-flags.hpp"
#include "codegen/topology/affinity_manager.hpp"
#include "codegen/util/parallel-context.hpp"
#include "codegen/util/profiling.hpp"
#include "codegen/util/timing.hpp"
#include "glo.hpp"
#include "interfaces/bench.hpp"
#include "scheduler/affinity_manager.hpp"
#include "scheduler/comm_manager.hpp"
#include "scheduler/topology.hpp"
#include "scheduler/worker.hpp"
#include "storage/column_store.hpp"
#include "storage/memory_manager.hpp"
#include "storage/table.hpp"

namespace harmonia {
// std::vector<std::thread> children;

// void *get_shm(std::string name, size_t size) {
//   int fd = shm_open(name.c_str(), O_CREAT | O_TRUNC | O_RDWR, 0666);

//   if (fd == -1) {
//     LOG(ERROR) << __FILE__ << ":" << __LINE__ << " " << strerror(errno)
//                << std::endl;
//     assert(false);
//   }

//   if (ftruncate(fd, size) < 0) {  //== -1){
//     LOG(ERROR) << __FILE__ << ":" << __LINE__ << " " << strerror(errno)
//                << std::endl;
//     assert(false);
//   }

//   void *mem = mmap(NULL, size, PROT_WRITE | PROT_READ, MAP_SHARED, fd, 0);
//   if (!mem) {
//     LOG(ERROR) << __FILE__ << ":" << __LINE__ << " " << strerror(errno)
//                << std::endl;
//     assert(false);
//   }

//   close(fd);
//   return mem;
// }

// [[noreturn]] void kill_orphans_and_widows(int s) {
//   for (auto &pid : children) {
//     kill(pid, SIGTERM);
//   }
//   exit(1);
// }

// void handle_child_termination(int sig) {
//   pid_t p;
//   int status;

//   while ((p = waitpid(-1, &status, WNOHANG)) > 0) {
//     LOG(INFO) << "Process " << p << " stopped or died.";
//     exit(-1);
//   }
// }

// void register_handler() {
//   signal(SIGINT, kill_orphans_and_widows);

//   //   {
//   //     struct sigaction sa {};
//   // #pragma clang diagnostic push
//   // #pragma clang diagnostic ignored "-Wdisabled-macro-expansion"
//   //     sa.sa_handler = handle_child_termination;
//   // #pragma clang diagnostic pop
//   //     sigaction(SIGCHLD, &sa, NULL);
//   //   }
// }
}  // namespace harmonia

#endif
