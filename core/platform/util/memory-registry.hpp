/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2014
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

#ifndef MEMORY_REGISTRY_HPP_
#define MEMORY_REGISTRY_HPP_

#include <cstdlib>

class MemoryRegistry {
 public:
  MemoryRegistry() = default;
  MemoryRegistry(const MemoryRegistry &) = default;
  MemoryRegistry(MemoryRegistry &&) = default;
  MemoryRegistry &operator=(const MemoryRegistry &) = default;
  MemoryRegistry &operator=(MemoryRegistry &&) = default;
  virtual ~MemoryRegistry() = default;

  virtual void reg(const void *mem, size_t bytes) = 0;
  virtual void unreg(const void *mem) = 0;
};

#endif /* MEMORY_REGISTRY_HPP_ */
