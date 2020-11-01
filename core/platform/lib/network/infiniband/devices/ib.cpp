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

#include <infiniband/verbs.h>

#include <cassert>
#include <fstream>
#include <platform/common/error-handling.hpp>
#include <platform/network/infiniband/devices/ib.hpp>
#include <platform/util/glog.hpp>

#include "lib/network/infiniband/private/ib_impl.hpp"

static auto findLocalCPUNumaNode(ibv_device *ib_dev) {
  std::ifstream in{ib_dev->ibdev_path + std::string{"/device/numa_node"}};
  uint32_t id;
  in >> id;
  return id;
}

ib::ib(size_t id, ibv_device *ctx, construction_guard)
    : id(id),
      p_impl(std::make_unique<IBimpl>(ctx)),
      local_cpu(findLocalCPUNumaNode(p_impl->context->device)) {}

ib::~ib() = default;

std::vector<ib> ib::discover() {
  std::vector<ib> ibs;

  int dev_cnt;
  ibv_device **dev_list = ibv_get_device_list(&dev_cnt);
  if (dev_list == nullptr) {
    if (errno == ENOSYS) {
      LOG(ERROR) << "Error reading IB devices";
      dev_cnt = 0;
    } else {
      linux_run(dev_list);
    }
  }
  ibs.reserve(dev_cnt);
  for (size_t ib_indx = 0; ib_indx < dev_cnt; ++ib_indx) {
    ibv_device *ib_dev = dev_list[ib_indx];
    assert(ib_dev && "No IB devices detected");

    ibs.emplace_back(ib_indx, ib_dev, construction_guard{});
  }

  if (dev_list) ibv_free_device_list(dev_list);

  return ibs;
}

MemoryRegistry &ib::getMemRegistry() const { return *p_impl; }
IBimpl &ib::getIBImpl() const { return *p_impl; }

void ib::ensure_listening() const { p_impl->ensure_listening(); }

ib_addr ib::get_server(const std::string &server, uint16_t port) const {
  return p_impl->get_server(server, port);
}
ib_addr ib::get_client(uint16_t port) const { return p_impl->get_client(port); }

std::ostream &operator<<(std::ostream &out, const ib &ibv) {
  out << ibv_get_device_name(ibv.p_impl->context->device) << " 0x" << std::hex
      << ibv_get_device_guid(ibv.p_impl->context->device) << std::dec << " "
      << ibv.p_impl->context->device->dev_path << " "
      << ibv.p_impl->context->device->ibdev_path << " "
      << ibv.p_impl->context->device->dev_name;
  return out;
}
