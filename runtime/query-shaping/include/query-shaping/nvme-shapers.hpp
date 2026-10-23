/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2026
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
#ifndef PROTEUS_NVME_SHAPERS_HPP
#define PROTEUS_NVME_SHAPERS_HPP
#include <codegen/expressions/expressionTypes.hpp>
#include <olap/plan/catalog-parser.hpp>
#include <query-shaping/input-prefix-query-shaper.hpp>
#include <vector>

namespace proteus {

class CPUOnlyNVMeMorsel : public proteus::InputPrefixQueryShaper {
  [[nodiscard]] pg getPlugin() const override { return pg{"nvme-block"}; }
  [[nodiscard]] DeviceType getDevice() override { return DeviceType::CPU; }
  /**
   * @param relName in the legacy catalog, e.g. inputs/ssbm100/lineorder.csv
   * This is very very hacky dealing with types. We need to come back and fix
   * the catalog for this or create an actual file format
   */
 public:
  [[nodiscard]] RelBuilder scan(
      const std::string &relName,
      std::initializer_list<std::string> relAttrs) override;

  [[nodiscard]] std::string getRelName(const std::string &base) override {
    LOG(FATAL) << "N/A to NVMe shapers";
  }

  RelBuilder distribute_build(RelBuilder input) override {
    auto rel = input
                   .router(getDOP(), scan_rounter_slack, RoutingPolicy::LOCAL,
                           getDevice(), getAffinitizer())
                   .memmove(scan_memmove_slack, getDevice());

    if (getDevice() == DeviceType::GPU) rel = rel.to_gpu();

    return rel;
  }

 public:
  CPUOnlyNVMeMorsel(std::vector<std::string> input_dirs,
                    const std::string &catalog_path,
                    decltype(input_sizes) input_sizes, bool allowMoves,
                    size_t scan_memmove_slack, size_t scan_rounter_slack,
                    size_t slack)
      : InputPrefixQueryShaper("N/A", input_sizes, allowMoves, slack),
        input_dirs(sort_vector(
            input_dirs)),  // sort so we always iterate in the same order
        catalog_path(catalog_path),
        scan_memmove_slack(scan_memmove_slack),
        scan_rounter_slack(scan_rounter_slack) {}

 protected:
  std::vector<std::filesystem::path> getMdForAttribute(
      const std::string &relName, const std::string &attr) const;

  /**
   * gross and hacky helper to construct dangling attributes for a relation and
   * vector of requested attributes using the legacy catalogue
   */
  std::vector<DanglingAttr> constructDanglingAttrs(
      const std::string &relName,
      const std::vector<std::string> &relAttrs) const;

  const std::vector<std::string> input_dirs;
  const std::string catalog_path;  // the path to the legacy catalog file
 private:
  static std::vector<std::string> sort_vector(std::vector<std::string> vec) {
    std::sort(vec.begin(), vec.end());
    return vec;
  }

  /**
   * @param dirPath the path to the directory to list files in
   * @return a sorted vector of paths to regular files in the directory
   */
  static std::vector<std::filesystem::path> getSortedDirectoryFiles(
      const std::string &dirPath) {
    std::vector<std::filesystem::path> paths;
    for (const auto &entry : std::filesystem::directory_iterator(dirPath)) {
      if (std::filesystem::is_regular_file(entry.path())) {
        paths.push_back(entry.path());
      }
    }
    std::sort(paths.begin(), paths.end());
    return paths;
  }
  size_t scan_memmove_slack;
  size_t scan_rounter_slack;
};

class GPUOnlyNVMeMorsel : public proteus::CPUOnlyNVMeMorsel {
  using proteus::CPUOnlyNVMeMorsel::CPUOnlyNVMeMorsel;
  [[nodiscard]] DeviceType getDevice() override { return DeviceType::GPU; }

  RelBuilder distribute_build(RelBuilder input) override {
    auto rel = input.router(
        DegreeOfParallelism(topology::getInstance().getGpuCount() * 12), 2,
        RoutingPolicy::LOCAL, DeviceType::CPU, getAffinitizer());

    if (doMove()) rel = rel.memmove(4, getDevice());
    rel = rel.router(getDOP(), 8, RoutingPolicy::LOCAL, DeviceType::CPU,
                     getAffinitizer());

    if (getDevice() == DeviceType::GPU) rel = rel.to_gpu();

    return rel;
  }

  /**
   * For GPUS, we do scan -> router (fanout) -> memmove -> router (fanin to GPU
   * DOP) because the asynchronous cufile API does not appear to scale either
   * with the number of threads or streams. On a A40 with 16x PCIe 4.0 we
   * achieve at best ~60% of the pcie bandwidth.
   * Our solution is to use the synchronous API, but have multiple memove
   * instances for each GPU so that we can have multiple IOs in flight to
   * achieve bandwidth. This approach achieves the maximum practical PCIe
   * bandwidth.
   * We should revisit the async API in future cuda versions, in theory it
   * should be more efficient
   */
  RelBuilder distribute_probe(RelBuilder input) override {
    auto rel = input.router(
        DegreeOfParallelism(topology::getInstance().getGpuCount() * 16), 2,
        RoutingPolicy::LOCAL, DeviceType::CPU, getAffinitizer());

    if (doMove()) rel = rel.memmove(16, getDevice());
    rel = rel.router(getDOP(), 8, RoutingPolicy::LOCAL, DeviceType::CPU,
                     getAffinitizer());

    if (getDevice() == DeviceType::GPU) rel = rel.to_gpu();

    return rel;
  }
};

};  // namespace proteus

#endif  // PROTEUS_NVME_SHAPERS_HPP
