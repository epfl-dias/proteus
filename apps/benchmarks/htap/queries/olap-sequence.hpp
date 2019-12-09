/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2019
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

#ifndef PROTEUS_OLAP_SEQUENCE_HPP
#define PROTEUS_OLAP_SEQUENCE_HPP

#include <include/plan/prepared-statement.hpp>
#include <topology/device-types.hpp>
#include <topology/topology.hpp>
#include <vector>

class OLAPSequence {
 private:
  std::vector<PreparedStatement> stmts;

 public:
  template <typename plugin_t>
  struct wrapper_t {};

  template <typename plugin_t>
  OLAPSequence(wrapper_t<plugin_t>, int client_id,
               const topology::cpunumanode &numa_node,
               const topology::cpunumanode &oltp_node,
               DeviceType dev = DeviceType::GPU);

  void run(int client_id, const topology::cpunumanode &numa_node,
           size_t repeat);
};

#endif /* PROTEUS_OLAP_SEQUENCE_HPP */
