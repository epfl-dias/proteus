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

#include "olap-sequence.hpp"

#include <aeolus-plugin.hpp>
#include <include/routing/affinitizers.hpp>
#include <include/routing/degree-of-parallelism.hpp>
#include <numeric>
#include <topology/affinity_manager.hpp>
#include <topology/topology.hpp>
#include <util/timing.hpp>

#include "../htap-cli-flags.hpp"
#include "ch/ch-queries.hpp"

template <typename plugin_t>
OLAPSequence::OLAPSequence(OLAPSequence::wrapper_t<plugin_t>, int client_id,
                           // const topology::cpunumanode &numa_node,
                           // const topology::cpunumanode &oltp_node,
                           exec_nodes olap_nodes, exec_nodes oltp_nodes,
                           DeviceType dev) {
  //    time_block t("TcodegenTotal_: ");
  //
  std::vector<SpecificCpuCoreAffinitizer::coreid_t> coreids;

  if (dev == DeviceType::CPU) {
    uint j = 0;

    for (auto &olap_n : olap_nodes) {
      for (auto id :
           (dynamic_cast<topology::cpunumanode *>(olap_n))->local_cores) {
        if (FLAGS_trade_core && FLAGS_elastic > 0 && j < FLAGS_elastic) {
          j++;
          continue;
        }
        coreids.emplace_back(id);
      }
    }

    if (FLAGS_elastic > 0) {
      uint i = 0;

      for (auto &oltp_n : oltp_nodes) {
        for (auto id :
             (dynamic_cast<topology::cpunumanode *>(oltp_n))->local_cores) {
          coreids.emplace_back(id);
          if (++i >= FLAGS_elastic) {
            break;
          }
        }

        if (i >= FLAGS_elastic) {
          break;
        }
      }
    }

    // {
    //   for (const auto &n : topology::getInstance().getCpuNumaNodes()) {
    //     if (n != numa_node) {
    //       for (size_t i = 0; i < std::min(4, n.local_cores.size()); ++i) {
    //         coreids.emplace_back(n.local_cores[i]);
    //       }
    //     }
    //   }
    // }

    DegreeOfParallelism dop{coreids.size()};

    auto aff_parallel = [&]() -> std::unique_ptr<Affinitizer> {
      return std::make_unique<SpecificCpuCoreAffinitizer>(coreids);
    };
  }
  DegreeOfParallelism dop{(dev == DeviceType::CPU) ? coreids.size()
                                                   : olap_nodes.size()};

  auto aff_parallel = [&]() -> std::unique_ptr<Affinitizer> {
    if (dev == DeviceType::CPU) {
      return std::make_unique<SpecificCpuCoreAffinitizer>(coreids);
    } else {
      return std::make_unique<GPUAffinitizer>();
    }
  };

  auto aff_reduce = []() -> std::unique_ptr<Affinitizer> {
    return std::make_unique<CpuCoreAffinitizer>();
  };

  typedef decltype(aff_parallel) aff_t;
  typedef decltype(aff_reduce) red_t;

  for (const auto &q : {
           Q<1>::prepare<plugin_t, aff_t, red_t>
           // ,Q<6>::prepare<plugin_t, aff_t, red_t>
       }) {
    stmts.emplace_back(q(dop, aff_parallel, aff_reduce, dev));
  }
}

std::ostream &operator<<(std::ostream &out, const OLAPSequence &seq) {
  out << "OLAP Stats" << std::endl;
  for (const auto &r : seq.stats) {
    out << "Sequence Run # " << r->run_id << std::endl;

    // per query avg time
    // std::vector<double> q_avg_time(r->sts.size(), 0.0);
    uint i = 0;
    for (const auto &q : r->sts) {
      out << "\t\tQuery # " << (i + 1) << " -";
      for (const auto &q_ts : q) {
        // q_avg_time[i] += q_ts;

        out << "\t" << q_ts;
      }
      out << std::endl;
      // q_avg_time[i] /= q.size();
      // out << "\t\tQuery (Avg) # " << (i + 1) << " - " << q_avg_time[i] << "
      // ms"
      //     << std::endl;
      i++;
    }

    // average sequence time
    // double average = (double)(std::accumulate(r->sequence_time.begin(),
    //                                           r->sequence_time.end(), 0.0)) /
    //                  ((double)r->sequence_time.size());

    // out << "\t\tQuery Sequence - " << average << " ms" << std::endl;

    out << "\tSequence Time "
        << " -";
    for (const auto &st : r->sequence_time) {
      out << "\t" << st;
    }
    out << std::endl;
  }
  return out;
}

void OLAPSequence::run(size_t repeat) {
  // TODO: Make affinity deterministic
  // exec_location{numa_node}.activate();
  static uint run = 0;

  OLAPSequenceStats *stats_local =
      new OLAPSequenceStats(++run, stmts.size(), repeat);

  LOG(INFO) << "Warm-up execution - BEGIN";
  stmts[0].execute();
  LOG(INFO) << "Warm-up execution - END";

  LOG(INFO) << "Exeucting OLAP Sequence[Client # " << this->client_id << "]";

  for (size_t i = 0; i < repeat; i++) {
    timepoint_t start_seq = std::chrono::system_clock::now();

    for (size_t j = 0; j < stmts.size(); j++) {
      timepoint_t start = std::chrono::system_clock::now();

      LOG(INFO) << stmts[j].execute();

      timepoint_t end = std::chrono::system_clock::now();

      stats_local->sts[j].push_back(
          std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
              .count());
    }

    timepoint_t end_seq = std::chrono::system_clock::now();

    stats_local->sequence_time.push_back(
        std::chrono::duration_cast<std::chrono::milliseconds>(end_seq -
                                                              start_seq)
            .count());
  }

  stats.push_back(stats_local);
}

template OLAPSequence::OLAPSequence(OLAPSequence::wrapper_t<AeolusLocalPlugin>,
                                    int client_id, exec_nodes olap_nodes,
                                    exec_nodes oltp_nodes, DeviceType dev);

template OLAPSequence::OLAPSequence(OLAPSequence::wrapper_t<AeolusRemotePlugin>,
                                    int client_id, exec_nodes olap_nodes,
                                    exec_nodes oltp_nodes, DeviceType dev);

template OLAPSequence::OLAPSequence(
    OLAPSequence::wrapper_t<AeolusElasticPlugin>, int client_id,
    exec_nodes olap_nodes, exec_nodes oltp_nodes, DeviceType dev);

template OLAPSequence::OLAPSequence(OLAPSequence::wrapper_t<AeolusCowPlugin>,
                                    int client_id, exec_nodes olap_nodes,
                                    exec_nodes oltp_nodes, DeviceType dev);

template OLAPSequence::OLAPSequence(OLAPSequence::wrapper_t<BinaryBlockPlugin>,
                                    int client_id, exec_nodes olap_nodes,
                                    exec_nodes oltp_nodes, DeviceType dev);
