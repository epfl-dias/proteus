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

#include "plan/prepared-statement.hpp"

#include "plan/plan-parser.hpp"
#include "topology/affinity_manager.hpp"
#include "topology/topology.hpp"

#if __has_include("ittnotify.h")
#include <ittnotify.h>
#else
#define __itt_resume() ((void)0)
#define __itt_pause() ((void)0)
#endif

static constexpr auto defaultCatalogJSON = "inputs";

void PreparedStatement::execute(bool deterministic_affinity) {
  auto &topo = topology::getInstance();

  // just to be sure...
  for (const auto &gpu : topology::getInstance().getGpus()) {
    set_device_on_scope d{gpu};
    gpu_run(cudaDeviceSynchronize());
  }

  for (const auto &gpu : topo.getGpus()) {
    set_exec_location_on_scope d{gpu};
    gpu_run(cudaProfilerStart());
  }
  __itt_resume();

  if (deterministic_affinity) {
    // Make affinity deterministic
    if (topo.getGpuCount() > 0) {
      exec_location{topo.getGpus()[0]}.activate();
    } else {
      exec_location{topo.getCpuNumaNodes()[0]}.activate();
    }
  }

  {
    time_block t("Texecute w sync: ");

    {
      time_block t("Texecute       : ");

      for (Pipeline *p : pipelines) {
        nvtxRangePushA("pip");
        {
          time_block t("T: ");

          p->open();
          p->consume(0);
          p->close();

          std::cout << std::dec;
        }
        nvtxRangePop();
      }

      std::cout << std::dec;
    }

    // just to be sure...
    for (const auto &gpu : topology::getInstance().getGpus()) {
      set_device_on_scope d{gpu};
      gpu_run(cudaDeviceSynchronize());
    }
  }

  __itt_pause();
  for (const auto &gpu : topo.getGpus()) {
    set_device_on_scope d{gpu};
    gpu_run(cudaProfilerStop());
  }
}

PreparedStatement PreparedStatement::from(const std::string &planPath,
                                          const std::string &label) {
  return from(planPath, label, defaultCatalogJSON);
}

PreparedStatement PreparedStatement::from(const std::string &planPath,
                                          const std::string &label,
                                          const std::string &catalogJSON) {
  {
    Catalog *catalog = &Catalog::getInstance();
    CachingService *caches = &CachingService::getInstance();
    catalog->clear();
    caches->clear();
  }

  std::vector<Pipeline *> pipelines;
  {
    time_block t("Tcodegen: ");

    ParallelContext *ctx = new ParallelContext(label, false);
    CatalogParser catalog{catalogJSON.c_str(), ctx};
    auto label_ptr = new std::string{label};
    PlanExecutor exec{planPath.c_str(), catalog, label_ptr->c_str(), ctx};

    ctx->compileAndLoad();

    return {ctx->getPipelines()};
  }
}
