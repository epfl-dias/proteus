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

#include "storage-load-policy-registry.hpp"

namespace proteus {

void StorageLoadPolicyRegistry::setDefaultLoader(Loader ld) {
  std::lock_guard<std::mutex> lock{m};
  defaultLoader = std::move(ld);
}

void StorageLoadPolicyRegistry::setLoader(std::string fileName, Loader ld) {
  std::lock_guard<std::mutex> lock{m};
  loaders.emplace(std::move(fileName), std::move(ld));
}

void StorageLoadPolicyRegistry::dropAllCustomLoaders() {
  std::lock_guard<std::mutex> lock{m};
  loaders.clear();
}

StorageLoadPolicyRegistry::StorageLoadPolicyRegistry(Loader defaultLoader)
    : defaultLoader(std::move(defaultLoader)) {}

StorageLoadPolicyRegistry::Loader StorageLoadPolicyRegistry::at(
    const std::string &fileName) {
  std::lock_guard<std::mutex> lock{m};
  try {
    return loaders.at(fileName);
  } catch (const std::out_of_range &) {
    return defaultLoader;
  }
}

}  // namespace proteus
