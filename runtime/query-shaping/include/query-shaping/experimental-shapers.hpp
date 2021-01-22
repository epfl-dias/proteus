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

  [[nodiscard]] RelBuilder collect(RelBuilder input) override;
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

  [[nodiscard]] RelBuilder collect(RelBuilder input) override;
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

  [[nodiscard]] RelBuilder collect(RelBuilder input) override;
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

class GPUOnlySingleSever : public proteus::InputPrefixQueryShaper {
  using proteus::InputPrefixQueryShaper::InputPrefixQueryShaper;
};

class LazyGPUOnlySingleSever : public GPUOnlySingleSever {
  using GPUOnlySingleSever::GPUOnlySingleSever;

 protected:
  [[nodiscard]] bool doMove() override;
  [[nodiscard]] int getSlack() override;
  [[nodiscard]] int getSlackReduce() override;

  [[nodiscard]] RelBuilder collect(RelBuilder input) override;
};

class CPUOnlySingleSever : public proteus::InputPrefixQueryShaper {
  using proteus::InputPrefixQueryShaper::InputPrefixQueryShaper;

 protected:
  [[nodiscard]] DeviceType getDevice() override;

  std::unique_ptr<Affinitizer> getAffinitizer() override;
};

class GPUOnlyHalfFile : public proteus::InputPrefixQueryShaper {
  using proteus::InputPrefixQueryShaper::InputPrefixQueryShaper;

 protected:
  [[nodiscard]] RelBuilder scan(
      const std::string& relName,
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
