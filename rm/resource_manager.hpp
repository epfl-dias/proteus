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

#ifndef RESOURCE_MANAGER_HPP_
#define RESOURCE_MANAGER_HPP_

#include "topology/topology.hpp"
#include <iostream>
#include <map>
#include <string>

namespace RM {

enum ENGINE_TYPE { OLTP, OLAP };

class ExecutorEngine {
public:
  ENGINE_TYPE type;
  std::vector<cpunumanode> numa_nodes;
  std::vector<core> cpu_cores;
  std::vector<gpunode> gpus;

  ExecutorEngine(ENGINE_TYPE type, cpunumanode &numa_node);
};

class ResourceManager {

protected:
public:
  // Singleton
  static ResourceManager &getInstance() {
    static ResourceManager instance;
    return instance;
  }

  // Prevent copies
  ResourceManager(const ResourceManager &) = delete;
  void operator=(const ResourceManager &) = delete;

  ResourceManager(ResourceManager &&) = delete;
  ResourceManager &operator=(ResourceManager &&) = delete;

  void init();
  void shutdown();

private:
  std::map<std::string, struct mem_file> mappings;

  ResourceManager() {}
  ~ResourceManager();
};

} // namespace RM

#endif /* RESOURCE_MANAGER_HPP_ */
