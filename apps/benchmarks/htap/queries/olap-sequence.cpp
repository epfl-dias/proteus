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
#include <topology/affinity_manager.hpp>
#include <topology/topology.hpp>
#include <util/timing.hpp>

#include "../htap-cli-flags.hpp"
#include "ch/ch-queries.hpp"

template <typename plugin_t>
OLAPSequence::OLAPSequence(OLAPSequence::wrapper_t<plugin_t>, int client_id,
                           const topology::cpunumanode &numa_node,
                           const topology::cpunumanode &oltp_node,
                           DeviceType dev) {
  //    time_block t("TcodegenTotal_: ");
  //
  std::vector<SpecificCpuCoreAffinitizer::coreid_t> coreids;

  if (dev == DeviceType::CPU) {
    uint j = 0;
    for (auto id : numa_node.local_cores) {
      if (FLAGS_trade_core && FLAGS_elastic > 0 && j < FLAGS_elastic) {
        j++;
        continue;
      }
      coreids.emplace_back(id);
    }

    if (FLAGS_elastic > 0) {
      uint i = 0;
      for (auto id : oltp_node.local_cores) {
        coreids.emplace_back(id);
        if (++i >= FLAGS_elastic) {
          break;
        }
      }

      if (FLAGS_trade_core) {
        for (auto id : numa_node.local_cores) {
          coreids.emplace_back(id);
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
  DegreeOfParallelism dop{(dev == DeviceType::CPU)
                              ? coreids.size()
                              : topology::getInstance().getGpuCount()};

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

  // using plugin_t = AeolusLocalPlugin;
  for (const auto &q : {Q<1>::prepare<plugin_t, aff_t, red_t>,
                        Q<6>::prepare<plugin_t, aff_t, red_t>}) {
    stmts.emplace_back(q(dop, aff_parallel, aff_reduce, dev));
  }
}

void OLAPSequence::run(int client_id, const topology::cpunumanode &numa_node,
                       size_t repeat) {
  // Make affinity deterministic
  exec_location{numa_node}.activate();

  LOG(INFO) << "taststas";
  stmts[0].execute();
  LOG(INFO) << "taststas";

  {
    time_block t("T_OLAP: ");
    LOG(INFO) << "taststas";

    for (size_t i = 0; i < repeat; i++) {
      for (auto &q : stmts) {
        LOG(INFO) << q.execute();
      }
    }
  }
}

template OLAPSequence::OLAPSequence(OLAPSequence::wrapper_t<AeolusLocalPlugin>,
                                    int client_id,
                                    const topology::cpunumanode &numa_node,
                                    const topology::cpunumanode &oltp_node,
                                    DeviceType dev);

template OLAPSequence::OLAPSequence(OLAPSequence::wrapper_t<AeolusRemotePlugin>,
                                    int client_id,
                                    const topology::cpunumanode &numa_node,
                                    const topology::cpunumanode &oltp_node,
                                    DeviceType dev);

template OLAPSequence::OLAPSequence(
    OLAPSequence::wrapper_t<AeolusElasticPlugin>, int client_id,
    const topology::cpunumanode &numa_node,
    const topology::cpunumanode &oltp_node, DeviceType dev);

template OLAPSequence::OLAPSequence(OLAPSequence::wrapper_t<AeolusCowPlugin>,
                                    int client_id,
                                    const topology::cpunumanode &numa_node,
                                    const topology::cpunumanode &oltp_node,
                                    DeviceType dev);

template OLAPSequence::OLAPSequence(OLAPSequence::wrapper_t<BinaryBlockPlugin>,
                                    int client_id,
                                    const topology::cpunumanode &numa_node,
                                    const topology::cpunumanode &oltp_node,
                                    DeviceType dev);
