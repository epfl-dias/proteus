/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2021
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

#ifndef PROTEUS_STORAGE_LOAD_POLICY_REGISTRY_HPP
#define PROTEUS_STORAGE_LOAD_POLICY_REGISTRY_HPP

#include <map>
#include <mutex>
#include <platform/storage/storage-manager.hpp>
#include <string>

namespace proteus {

class StorageLoadPolicyRegistry {
  using Loader = std::function<FileRecord(StorageManager &, const std::string &,
                                          size_t typeSize)>;

  std::mutex m;
  std::map<std::string, Loader> loaders;
  Loader defaultLoader;

 public:
  StorageLoadPolicyRegistry(Loader defaultLoader);
  void setDefaultLoader(Loader ld);
  void setLoader(std::string, Loader ld);
  void dropAllCustomLoaders();

  Loader at(const std::string &fileName);
};

}  // namespace proteus

#endif /* PROTEUS_STORAGE_LOAD_POLICY_REGISTRY_HPP */
