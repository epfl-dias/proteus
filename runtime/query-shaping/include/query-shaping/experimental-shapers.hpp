/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2021
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

#ifndef PROTEUS_EXPERIMENTAL_SHAPERS_HPP
#define PROTEUS_EXPERIMENTAL_SHAPERS_HPP

#include <query-shaping/nvme-shapers.hpp>
#include <query-shaping/scale-out-query-shaper.hpp>

namespace proteus {

class CPUOnlyShuffleAll : public proteus::ScaleOutQueryShaper {
  using proteus::ScaleOutQueryShaper::ScaleOutQueryShaper;

 protected:
  std::unique_ptr<Affinitizer> getAffinitizer() override;

  [[nodiscard]] RelBuilder distribute_probe_interserver(
      RelBuilder input) override;
};

class GPUOnlyShuffleAll : public proteus::ScaleOutQueryShaper {
  using proteus::ScaleOutQueryShaper::ScaleOutQueryShaper;

 protected:
  [[nodiscard]] DeviceType getDevice() override;
  [[nodiscard]] RelBuilder distribute_probe_interserver(
      RelBuilder input) override;

  [[nodiscard]] RelBuilder distribute_probe_intraserver(
      RelBuilder input) override;

  [[nodiscard]] RelBuilder collect_packed(RelBuilder input) override;
};

class CPUOnlyShuffleAllCorrectPlan : public proteus::ScaleOutQueryShaper {
  using proteus::ScaleOutQueryShaper::ScaleOutQueryShaper;

 protected:
  std::unique_ptr<Affinitizer> getAffinitizer() override;

  [[nodiscard]] DeviceType getDevice() override;
  [[nodiscard]] RelBuilder distribute_probe_interserver(
      RelBuilder input) override;
  [[nodiscard]] RelBuilder distribute_probe_intraserver(
      RelBuilder input) override;
};

class GPUOnlyShuffleAllCorrectPlan : public proteus::ScaleOutQueryShaper {
  using proteus::ScaleOutQueryShaper::ScaleOutQueryShaper;

 protected:
  std::unique_ptr<Affinitizer> getAffinitizer() override;

  [[nodiscard]] DeviceType getDevice() override;
  [[nodiscard]] RelBuilder distribute_probe_interserver(
      RelBuilder input) override;

  [[nodiscard]] RelBuilder distribute_probe_intraserver(
      RelBuilder input) override;

  [[nodiscard]] RelBuilder collect_packed(RelBuilder input) override;

  friend class HybridShuffleAllCorrectPlan;
};

class CPUOnlyNoShuffle : public proteus::ScaleOutQueryShaper {
  using proteus::ScaleOutQueryShaper::ScaleOutQueryShaper;
};

class LazyGPUNoShuffle : public CPUOnlyNoShuffle {
  using CPUOnlyNoShuffle::CPUOnlyNoShuffle;

 protected:
  [[nodiscard]] bool doMove() override;
  [[nodiscard]] int getSlack() override;
  [[nodiscard]] DeviceType getDevice() override;

  [[nodiscard]] RelBuilder collect_packed(RelBuilder input) override;
};

class GPUOnlyLocalAllCorrectPlan : public GPUOnlyShuffleAllCorrectPlan {
  using GPUOnlyShuffleAllCorrectPlan::GPUOnlyShuffleAllCorrectPlan;

 protected:
  [[nodiscard]] RelBuilder distribute_probe_interserver(
      RelBuilder input) override;

  [[nodiscard]] RelBuilder distribute_probe_intraserver(
      RelBuilder input) override;
};

class LazyGPUOnlyLocalAllCorrectPlan : public GPUOnlyLocalAllCorrectPlan {
  using GPUOnlyLocalAllCorrectPlan::GPUOnlyLocalAllCorrectPlan;

 protected:
  [[nodiscard]] RelBuilder distribute_probe_intraserver(
      RelBuilder input) override;
};

class GPUOnlySingleServer : public proteus::InputPrefixQueryShaper {
  using proteus::InputPrefixQueryShaper::InputPrefixQueryShaper;
};

/**
 * Single-server GPU shaper for metadata-driven (md_dirs) scans.
 *
 * This keeps the "single-server" query-shaping behavior while using the
 * NVMe metadata scan path (relation_md -> NvmePlugin) so compressed datasets
 * are actually read from the configured md directories.
 */
class GPUOnlySingleServerCompressed : public proteus::CPUOnlyNVMeMorsel {
 public:
  GPUOnlySingleServerCompressed(
      std::vector<std::string> input_dirs, const std::string &catalog_path,
      decltype(input_sizes) input_sizes, bool allowMoves,
      size_t scan_memmove_slack, size_t scan_router_slack, size_t slack,
      bool emit_compressed_blocks)
      : CPUOnlyNVMeMorsel(std::move(input_dirs), catalog_path,
                          std::move(input_sizes), allowMoves,
                          scan_memmove_slack, scan_router_slack, slack),
        emit_compressed_blocks_(emit_compressed_blocks) {}

 protected:
  [[nodiscard]] DeviceType getDevice() override { return DeviceType::GPU; }

  [[nodiscard]] RelBuilder scan(
      const std::string &relName,
      std::initializer_list<std::string> relAttrs) override {
    std::vector<std::string> attrs_vec{relAttrs.begin(), relAttrs.end()};
    std::vector<DanglingAttr> attrs = constructDanglingAttrs(relName, attrs_vec);
    RecordType scan_record_type = rel(relName)(attrs);

    auto meta_data_records_map = scan_record_type.getArgsMap();
    std::vector<std::pair<RecordAttribute *, std::vector<std::filesystem::path>>>
        relation_md;
    relation_md.reserve(attrs_vec.size());
    for (const auto &attr : attrs_vec) {
      auto md_paths = getMdForAttribute(relName, attr);
      relation_md.emplace_back(meta_data_records_map[attr], md_paths);
    }

    auto builder = getBuilder();
    auto rel = builder.scan(
        relation_md,
        /*preload_to_gpu=*/true,
        /*emit_compressed_blocks=*/emit_compressed_blocks_);
    rel = rel.hintRowCount(getRowHint(relName));
    return rel;
  }

  [[nodiscard]] RelBuilder distribute_probe(RelBuilder input) override {
    auto rel = input.router(getDOP(), getSlack(), RoutingPolicy::LOCAL,
                            getDevice(), getAffinitizer());

    if (doMove()) rel = rel.memmove(getSlack(), getDevice());
    if (getDevice() == DeviceType::GPU) rel = rel.to_gpu();

    return rel;
  }

  [[nodiscard]] RelBuilder distribute_build(RelBuilder input) override {
    // Match GPUOnlySingleServer build semantics: broadcast build-side blocks
    // before routing by broadcast target. In this shaper we preload concrete
    // GPU pointers, so MemBroadcastDevice is valid.
    auto rel = input
                   .membrdcst(getDOP(), getDevice() == DeviceType::CPU, false)
                   .router(
                       [&](const auto &arg) -> expression_t {
                         return arg["__broadcastTarget"];
                       },
                       getDOP(), getSlack(), getDevice(), getAffinitizer());

    if (getDevice() == DeviceType::GPU) rel = rel.to_gpu();
    return rel;
  }

 private:
  bool emit_compressed_blocks_;
};

class LazyGPUOnlySingleServer : public GPUOnlySingleServer {
  using GPUOnlySingleServer::GPUOnlySingleServer;

 protected:
  [[nodiscard]] bool doMove() override;
  [[nodiscard]] int getSlack() override;
  [[nodiscard]] int getSlackReduce() override;

  [[nodiscard]] RelBuilder collect_packed(RelBuilder input) override;
};

class CPUOnlySingleServer : public proteus::InputPrefixQueryShaper {
  using proteus::InputPrefixQueryShaper::InputPrefixQueryShaper;

 protected:
  [[nodiscard]] DeviceType getDevice() override;

  std::unique_ptr<Affinitizer> getAffinitizer() override;
};

class CPUOnlySingleServerMorsel : public CPUOnlySingleServer {
  using CPUOnlySingleServer::CPUOnlySingleServer;

 public:
  [[nodiscard]] RelBuilder distribute_build(RelBuilder input) override;
};

class HybridSingleServer : public proteus::InputPrefixQueryShaper {
  using proteus::InputPrefixQueryShaper::InputPrefixQueryShaper;

 protected:
  [[nodiscard]] RelBuilder parallel(
      RelBuilder probe, const std::vector<RelBuilder> &builds,
      const std::function<RelBuilder(
          RelBuilder, const std::vector<RelBuilder> &)> &pathBuilder) override;
};

class LazyHybridSingleServer : public proteus::InputPrefixQueryShaper {
  using proteus::InputPrefixQueryShaper::InputPrefixQueryShaper;

 protected:
  [[nodiscard]] RelBuilder parallel(
      RelBuilder probe, const std::vector<RelBuilder> &builds,
      const std::function<RelBuilder(
          RelBuilder, const std::vector<RelBuilder> &)> &pathBuilder) override;
};

class LazyHybridSingleServerMorsel : public proteus::InputPrefixQueryShaper {
  using proteus::InputPrefixQueryShaper::InputPrefixQueryShaper;

 protected:
  [[nodiscard]] RelBuilder parallel(
      RelBuilder probe, const std::vector<RelBuilder> &builds,
      const std::function<RelBuilder(
          RelBuilder, const std::vector<RelBuilder> &)> &pathBuilder) override;
};

class HybridSingleServerMorsel : public proteus::InputPrefixQueryShaper {
  using proteus::InputPrefixQueryShaper::InputPrefixQueryShaper;

 protected:
  [[nodiscard]] RelBuilder parallel(
      RelBuilder probe, const std::vector<RelBuilder> &builds,
      const std::function<RelBuilder(
          RelBuilder, const std::vector<RelBuilder> &)> &pathBuilder) override;
};

class GPUOnlyHalfFile : public proteus::InputPrefixQueryShaper {
  using proteus::InputPrefixQueryShaper::InputPrefixQueryShaper;

 protected:
  [[nodiscard]] RelBuilder scan(
      const std::string &relName,
      std::initializer_list<std::string> relAttrs) override;
};

class CPUOnlyHalfFile : public GPUOnlyHalfFile {
  using GPUOnlyHalfFile::GPUOnlyHalfFile;

 protected:
  [[nodiscard]] DeviceType getDevice() override;

  [[nodiscard]] std::unique_ptr<Affinitizer> getAffinitizer() override;
};

}  // namespace proteus

#endif /* PROTEUS_EXPERIMENTAL_SHAPERS_HPP */
