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
#include "lib/util/flush-operator-tree.hpp"
#include "lib/util/jit/pipeline.hpp"
#include "plan-parser.hpp"

static constexpr auto defaultCatalogJSON = "inputs";

QueryResult PreparedStatement::execute() {
  std::vector<std::chrono::milliseconds> tlog;
  return execute(tlog);
}

static auto defaultTimeBlockFactory(const char *s) { return time_block(s); }

time_block PreparedStatement::SilentExecution(const char *s) {
  return time_block([](const auto &t) {});
}

QueryResult PreparedStatement::execute(
    std::function<time_block(const char *)> f) {
  std::vector<std::chrono::milliseconds> tlog;
  return execute(tlog, std::move(f));
}

QueryResult PreparedStatement::execute(
    std::vector<std::chrono::milliseconds> &tlog) {
  return execute(tlog, defaultTimeBlockFactory);
}

QueryResult PreparedStatement::execute(QueryParameters qs) {
  auto session =
      static_cast<QueryParameters *>(MemoryManager::mallocPinned(sizeof(qs)));
  *session = std::move(qs);

  std::vector<std::chrono::milliseconds> tlog;
  auto res = execute(tlog, defaultTimeBlockFactory, session);

  session->~QueryParameters();
  MemoryManager::freePinned(session);
  return res;
}

QueryResult PreparedStatement::execute(
    std::vector<std::chrono::milliseconds> &tlog,
    std::function<time_block(const char *)> f) {
  return execute(tlog, f, nullptr);
}

QueryResult PreparedStatement::execute(
    std::vector<std::chrono::milliseconds> &tlog,
    std::function<time_block(const char *)> f, const void *session) {
  bool freePtr = false;
  if (!session) {
    session = MemoryManager::mallocPinned(sizeof(size_t));
    freePtr = true;
  }
  bool deterministic_affinity = true;
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

  {
    time_block twsync = f("Texecute w sync: ");

    {
      time_block texecute = f("Texecute       : ");

      for (auto &p : pipelines) {
        nvtxRangePushA("pip");
        {
          time_block t2([&](const auto &tmil) { tlog.emplace_back(tmil); });
          time_block t = f("T: ");

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

  profiling::pause();

  if (freePtr) MemoryManager::freePinned(const_cast<void *>(session));

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

    return exec.builtPlan.prepare();
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

    return exec.builtPlan.prepare();
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
    std::vector<std::unique_ptr<Pipeline>> pips, std::string outputFile,
    std::shared_ptr<Operator> planRoot)
    : pipelines(uniqueToShared(std::move(pips))),
      outputFile(std::move(outputFile)),
      planRoot(std::move(planRoot)) {}

std::ostream &operator<<(std::ostream &out, const PreparedStatement &p) {
  return out << *p.planRoot;
}
