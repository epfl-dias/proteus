/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2020
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

#ifndef PROTEUS_REMOTE_MANAGED_POINTER_HPP
#define PROTEUS_REMOTE_MANAGED_POINTER_HPP

#include <platform/memory/maganed-pointer.hpp>

using server_id_t = uint64_t;

namespace proteus {
template <typename T>
class remote_managed {
  proteus::managed_ptr ptr;
  server_id_t server;

 public:
  remote_managed(decltype(ptr) ptr, decltype(server) server)
      : ptr(std::move(ptr)), server(server) {}

  remote_managed(nullptr_t) : remote_managed(nullptr, 0) {}

  void *get() { return ptr.get(); }

  auto release() { return ptr.release(); }

  [[nodiscard]] explicit operator bool() const { return (bool)ptr; }
};

using remote_managed_ptr = remote_managed<void>;
}  // namespace proteus

#endif  // PROTEUS_REMOTE_MANAGED_POINTER_HPP
