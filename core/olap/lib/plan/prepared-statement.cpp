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

#include <olap/plan/prepared-statement.hpp>
#include <olap/util/parallel-context.hpp>
#include <platform/memory/memory-manager.hpp>
#include <platform/topology/affinity_manager.hpp>
#include <platform/topology/topology.hpp>
#include <platform/util/profiling.hpp>
#include <platform/util/timing.hpp>
#include <utility>

#include "lib/util/caching.hpp"
#include "lib/util/catalog.hpp"
#include "lib/util/jit/pipeline.hpp"
#include "plan-parser.hpp"

static constexpr auto defaultCatalogJSON = "inputs";

QueryResult PreparedStatement::execute(bool deterministic_affinity) {
  auto &topo = topology::getInstance();

  // just to be sure...
  for (const auto &gpu : topology::getInstance().getGpus()) {
    set_device_on_scope d{gpu};
    gpu_run(cudaDeviceSynchronize());
  }

  profiling::resume();

  if (deterministic_affinity) {
    // Make affinity deterministic
    if (topo.getGpuCount() > 0) {
      exec_location{topo.getGpus()[0]}.activate();
    } else {
      exec_location{topo.getCpuNumaNodes()[0]}.activate();
    }
  }

  static uint32_t year = 1993;
  void *session = MemoryManager::mallocPinned(sizeof(int64_t));
  *static_cast<uint32_t *>(session) = ++year;

  {
    time_block t("Texecute w sync: ");

    {
      time_block t("Texecute       : ");

      for (auto &p : pipelines) {
        nvtxRangePushA("pip");
        {
          time_block t("T: ");

          p->open(session);
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

  MemoryManager::freePinned(session);

  profiling::pause();

  return {outputFile};
}

PreparedStatement PreparedStatement::from(const std::string &planPath,
                                          const std::string &label) {
  return from(planPath, label, defaultCatalogJSON);
}

PreparedStatement PreparedStatement::from(
    const std::span<const std::byte> &plan, const std::string &label) {
  return from(plan, label, defaultCatalogJSON);
}

PreparedStatement PreparedStatement::from(
    const std::string &planPath, const std::string &label,
    std::unique_ptr<AffinitizationFactory> affFactory) {
  {
    Catalog *catalog = &Catalog::getInstance();
    CachingService *caches = &CachingService::getInstance();
    catalog->clear();
    caches->clear();
  }

  {
    time_block t("Tcodegen: ");

    auto ctx = new ParallelContext(label, false);
    CatalogParser catalog{defaultCatalogJSON, ctx};
    auto label_ptr = new std::string{label};
    PlanExecutor exec{planPath.c_str(), catalog, std::move(affFactory),
                      label_ptr->c_str()};

    exec.compileAndLoad();

    return {exec.ctx->getPipelines(), exec.ctx->getModuleName()};
  }
}

PreparedStatement PreparedStatement::from(const std::string &planPath,
                                          const std::string &label,
                                          const std::string &catalogJSON) {
  return from(mmap_file{planPath, PAGEABLE}.asSpan(), label, catalogJSON);
}

PreparedStatement PreparedStatement::from(
    const std::span<const std::byte> &planPath, const std::string &label,
    const std::string &catalogJSON) {
  {
    Catalog *catalog = &Catalog::getInstance();
    CachingService *caches = &CachingService::getInstance();
    catalog->clear();
    caches->clear();
  }

  {
    time_block t("Tcodegen: ");

    auto ctx = new ParallelContext(label, false);
    CatalogParser catalog{catalogJSON.c_str(), ctx};
    auto label_ptr = new std::string{label};
    PlanExecutor exec{planPath, catalog, label_ptr->c_str()};

    exec.compileAndLoad();

    return {exec.ctx->getPipelines(), exec.ctx->getModuleName()};
  }
}

std::vector<std::shared_ptr<Pipeline>> uniqueToShared(
    std::vector<std::unique_ptr<Pipeline>> pips) {
  std::vector<std::shared_ptr<Pipeline>> ret{
      std::make_move_iterator(pips.begin()),
      std::make_move_iterator(pips.end())};
  return ret;
}

PreparedStatement::PreparedStatement(
    std::vector<std::unique_ptr<Pipeline>> pips, std::string outputFile)
    : pipelines(uniqueToShared(std::move(pips))),
      outputFile(std::move(outputFile)) {}
