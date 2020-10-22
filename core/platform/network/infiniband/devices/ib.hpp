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

#ifndef PROTEUS_IB_HPP
#define PROTEUS_IB_HPP

#include <infiniband/verbs.h>

#include <memory>
#include <util/memory-registry.hpp>
#include <vector>

struct ib_addr {
  ibv_gid gid;
  uint16_t lid;
  uint32_t psn;
  uint32_t qpn;
};

class IBimpl;
struct ibv_device;

class ib {
 public:
  size_t id;

 private:
  std::unique_ptr<IBimpl> p_impl;

 public:
  // CPU socket id
  uint32_t local_cpu;

  class construction_guard {
   private:
    construction_guard() = default;
    friend class ib;
  };

  [[nodiscard]] IBimpl &getIBImpl() const;
  friend class IBHandler;

 public:
  ib(size_t id, ibv_device *device,
     // do not remove argument!!!
     construction_guard = {});

  // Move is only allowed because we place everything in a vector
  // to avoid an indirection level and vector requires move constructors
  ib(ib &&) noexcept = default;
  ib &operator=(ib &&) noexcept = default;

  ib(const ib &) = delete;
  ib &operator=(const ib &) = delete;

  ~ib();

  static std::vector<ib> discover();

  [[nodiscard]] MemoryRegistry &getMemRegistry() const;
  void ensure_listening() const;

  [[nodiscard]] ib_addr get_server(const std::string &server,
                                   uint16_t port) const;
  [[nodiscard]] ib_addr get_client(uint16_t port) const;

  friend std::ostream &operator<<(std::ostream &out, const ib &ibv);
};

#endif /* PROTEUS_IB_HPP */
