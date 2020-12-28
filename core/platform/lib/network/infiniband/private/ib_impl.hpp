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

#ifndef PROTEUS_IB_IMPL_HPP
#define PROTEUS_IB_IMPL_HPP

#include <infiniband/verbs.h>

#include <memory>
#include <platform/common/error-handling.hpp>
#include <platform/memory/block-manager.hpp>
#include <platform/util/memory-registry.hpp>

class IBimpl final : public MemoryRegistry {
 public:
  struct IbvContextClose {
    void operator()(ibv_context *b) { ibv_close_device(b); }
  };

  std::unique_ptr<ibv_context, IbvContextClose> context;
  ibv_pd *pd;
  ibv_comp_channel *comp_channel;
  ibv_cq *cq;
  ibv_qp *qp;
  ib_addr addr;
  decltype(ibv_qp_attr::port_num) ib_port;
  uint8_t ib_sl;
  uint8_t ib_gidx;

  std::map<const void *, ibv_mr *> reged_mem;
  std::mutex m_reg;

  [[nodiscard]] ib_addr get_server(const std::string &server,
                                   uint16_t port) const;
  [[nodiscard]] ib_addr get_client(uint16_t port) const;

  void post_recv();

  void ensure_listening();

  [[nodiscard]] ibv_device_attr query_device() const;

  [[nodiscard]] ibv_port_attr query_port(
      decltype(ibv_qp_attr::port_num) ibport) const;

  [[nodiscard]] ibv_gid query_port_gid(decltype(ibv_qp_attr::port_num) ibport,
                                       int ibgidx) const;

 public:
  explicit IBimpl(ibv_device *device);
  IBimpl(const IBimpl &) = delete;
  IBimpl(IBimpl &&) noexcept = delete;
  IBimpl &operator=(const IBimpl &) = delete;
  IBimpl &operator=(const IBimpl &&) noexcept = delete;

  ~IBimpl() override;

  void reg(const void *mem, size_t bytes) override;

  void unreg(const void *mem) override;
};

#endif /* PROTEUS_IB_IMPL_HPP */
