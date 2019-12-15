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

#include <deque>
#include <include/plan/prepared-statement.hpp>
#include <topology/device-types.hpp>
#include <topology/topology.hpp>
#include <vector>

using exec_nodes = std::vector<topology::numanode *>;
typedef std::chrono::time_point<std::chrono::system_clock> timepoint_t;

struct OLAPSequenceStats {
  const uint run_id;
  // runtimes[# queries][# runs] milliseconds
  std::vector<std::vector<size_t>> sts;

  std::vector<size_t> sequence_time;

  OLAPSequenceStats(uint run_id, size_t number_of_queries, size_t repeat)
      : run_id(run_id) {
    sts.reserve(number_of_queries);
    for (size_t i = 0; i < number_of_queries; i++) {
      sts.emplace_back(std::vector<size_t>());
      sts[i].reserve(repeat);
    }

    sequence_time.reserve(repeat);
  }
};

class OLAPSequence {
 private:
  std::vector<PreparedStatement> stmts;
  int client_id;

  exec_nodes olap_nodes;
  exec_nodes oltp_nodes;

  std::deque<OLAPSequenceStats *> stats;  // insert per run.

 public:
  template <typename plugin_t>
  struct wrapper_t {};

  template <typename plugin_t>
  OLAPSequence(wrapper_t<plugin_t>, int client_id, exec_nodes olap_nodes,
               exec_nodes oltp_nodes, DeviceType dev);

  void run(size_t repeat);

  friend std::ostream &operator<<(std::ostream &stream,
                                  const OLAPSequence &seq);

  // TODO: implement
  //~OLAPSequence()
};

std::ostream &operator<<(std::ostream &stream, const OLAPSequence &seq);

#endif /* PROTEUS_OLAP_SEQUENCE_HPP */
