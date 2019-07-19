/*
                  ..

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

#define OLTP_PATH "/scratch/raza/htap/opt/aeolus/aeolus-server"
#define OLAP_PATH "./../pelago/proteusmain-server"

#define LAUNCH_OLTP false
#define LAUNCH_OLAP false

#include "cm/comm_manager.hpp"
#include "topology/topology.hpp"
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <signal.h>
#include <sys/types.h>
#include <thread>
#include <unistd.h>
#include <vector>

pid_t init_oltp_process();
pid_t init_olap_process();
void kill_orphans_and_widows(int s);
void register_handler();

std::vector<pid_t> children;

int main(int argc, char **argv) {

  const auto &vec = topology::getInstance().getCpuNumaNodes();

  // std::cout << vec[0].local_cpu_set << std::endl;

  std::cout << "------- HTAP  ------" << std::endl;

  std::cout << "\tInitializing communication manager..." << std::endl;
  CM::CommManager::getInstance().init();

  std::cout << "---------------------" << std::endl;

  std::cout << "[HTAP] HTAP Process ID: " << getppid() << std::endl;

#if LAUNCH_OLTP
  pid_t pid_oltp = init_oltp_process();
  assert(pid_oltp != 0);
  children.push_back(pid_oltp);
  std::cout << "[HTAP] OLTP Process ID: " << pid_oltp << std::endl;
#endif

#if LAUNCH_OLAP
  pid_t pid_olap = init_olap_process();
  assert(pid_olap != 0);
  children.push_back(pid_olap);
  std::cout << "[HTAP] OLAP Process ID: " << pid_olap << std::endl;
#endif

  // Register signal handler
  register_handler();

  while (true) {
    std::this_thread::sleep_for(std::chrono::seconds(60));
  }
}

pid_t init_oltp_process() {

  pid_t pid_oltp = fork();

  if (pid_oltp == 0) {
    execl("/scratch/raza/htap/opt/aeolus/aeolus-server", "");
    assert(false && "OLTP process call failed");

  } else {
    return pid_oltp;
  }
}

pid_t init_olap_process() { assert(false && "Unimplemented"); }

void kill_orphans_and_widows(int s) {
  std::cout << "[SIGNAL] Recieved signal: " << s << std::endl;
  for (auto &pid : children) {
    kill(pid, SIGTERM);
  }
  exit(1);
}

void register_handler() {
  // struct sigaction sigIntHandler;
  // sigIntHandler.sa_handler = kill_orphans_and_widows;
  // sigemptyset(&sigIntHandler.sa_mask);
  // sigIntHandler.sa_flags = 0;
  // sigaction(SIGINT, &sigIntHandler, NULL);
  signal(SIGINT, kill_orphans_and_widows);
}
